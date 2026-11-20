"""MiniMax H3 speed-ups as opt-in nodes, leaving ComfyUI core files untouched.

"MiniMax H3 Text Encoder BF16": put it after the CLIP loader. CLIPs that pass through it run the language model of
the MiniMax H3 text encoder (Qwen3-VL-32B) in bf16; every other CLIP is unchanged. ComfyUI's SDClipModel calls text
encoders with dtype=float32, so the hidden states stay fp32 and each nvfp4 layer is dequantized to fp32 for an fp32
GEMM. On ROCm (R9700) that lands on a small-tile fp32 kernel: ~32 s of a ~38 s encode. The weights file, the loader and
the vision tower are unchanged; the conditioning still leaves the encoder as fp32. The MiniMax H3 Image encoder
(PH3ImageQwen3VL) gets the same bf16 language model; its hidden states go back to fp32 before its fp32 text connector.

"Unpinned Offload": for the MODEL and/or CLIP passed through it, weights moved out of VRAM to make room for another
model are not page-locked (cudaHostRegister) in RAM. With the 15 GB text encoder and the 20 GB H3 Image DiT, which do
not fit together in 32 GB, every prompt change swaps ~8 GB each way and the pinning costs more than it saves.

"Load VAE (BF16)": a VAE loader that runs the VAE with bf16 activations instead of fp16. The MiniMax H3 int8 video
VAE then takes comfy-kitchen's tiled int8 GEMM (as the DiT does) instead of the fp16 WMMA kernel. With fast_rms_rope
on, the attention layers of that VAE (and no other model) use rmsrope.py's Triton RMSNorm+RoPE for head dim 64 instead
of comfy-kitchen's generic HIP kernel (~9x faster per call, closer to an fp32 reference). With fast_convrot_quant
on, int8 GEMM inputs quantized while that VAE decodes go through convrot.py's Triton ConvRot quantizer (~3x faster).
"""
import threading
import types

import torch
import comfy.model_patcher
import comfy.sd
from comfy.text_encoders import minimax

OPTION = "h3accel_te_bf16"
_state = threading.local()

# Both hooks below are inert unless the CLIP being encoded carries OPTION in its patcher's model_options.
_encode_from_tokens = comfy.sd.CLIP.encode_from_tokens


def _encode(self, *args, **kwargs):
    previous = getattr(_state, "bf16", False)
    _state.bf16 = bool(self.patcher.model_options.get(OPTION, False))
    try:
        return _encode_from_tokens(self, *args, **kwargs)
    finally:
        _state.bf16 = previous


_forward = minimax.MiniMaxQwen3VL.forward


def _lm_forward(self, input_ids, attention_mask=None, embeds=None, *args, dtype=None, **kwargs):
    # exact class only: the H3 Image encoder (PH3ImageQwen3VL) subclasses this one
    if getattr(_state, "bf16", False) and type(self) is minimax.MiniMaxQwen3VL and dtype == torch.float32:
        dtype = torch.bfloat16
        if embeds is not None:
            embeds = embeds.to(torch.bfloat16)
    return _forward(self, input_ids, attention_mask, embeds, *args, dtype=dtype, **kwargs)


_img_forward_orig = minimax.PH3ImageQwen3VL.forward


def _img_forward(self, input_ids, attention_mask=None, embeds=None, *args, dtype=None, **kwargs):
    if getattr(_state, "bf16", False) and type(self) is minimax.PH3ImageQwen3VL and dtype == torch.float32:
        if embeds is not None:
            embeds = embeds.to(torch.bfloat16)
        x, hidden_states = _forward(self, input_ids, attention_mask, embeds, *args, dtype=torch.bfloat16, **kwargs)
        return x.float(), self.text_connector(hidden_states.float())
    return _img_forward_orig(self, input_ids, attention_mask, embeds, *args, dtype=dtype, **kwargs)


comfy.sd.CLIP.encode_from_tokens = _encode
minimax.MiniMaxQwen3VL.forward = _lm_forward
minimax.PH3ImageQwen3VL.forward = _img_forward

NO_PIN = "h3accel_no_pin"
_pin_weight_to_device = comfy.model_patcher.ModelPatcher.pin_weight_to_device


def _pin(self, key):
    # inert unless this patcher (or the patcher it was cloned from) went through the Unpinned Offload node
    if self.model_options.get(NO_PIN, False):
        return
    return _pin_weight_to_device(self, key)


comfy.model_patcher.ModelPatcher.pin_weight_to_device = _pin


class MiniMaxH3TextEncoderBF16:
    DESCRIPTION = ("Runs the MiniMax H3 (video and image) text encoder's language model in bf16 instead of fp32: same "
                   "weights file, ~4x faster encode on AMD. Has no effect on other text encoders.")
    CATEGORY = "advanced/model"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "enabled": ("BOOLEAN", {"default": True})}}

    def apply(self, clip, enabled):
        out = clip.clone()
        out.patcher.model_options[OPTION] = enabled
        return (out,)


class MiniMaxH3VAELoaderBF16:
    DESCRIPTION = ("Loads a VAE with bf16 activations instead of the default fp16. For the MiniMax H3 int8 video VAE this "
                   "puts its int8 GEMMs on comfy-kitchen's tiled kernel (the one the DiT uses), 8-17 % faster per call. "
                   "fast_rms_rope: Triton RMSNorm+RoPE for the H3 VAE's attention (head dim 64). fast_convrot_quant: Triton "
                   "ConvRot int8 quantizer for this VAE's decode.")
    CATEGORY = "loaders"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"vae_name": (folder_paths.get_filename_list("vae"),),
                             "fast_rms_rope": ("BOOLEAN", {"default": True}),
                             "fast_convrot_quant": ("BOOLEAN", {"default": True})}}

    def load(self, vae_name, fast_rms_rope=True, fast_convrot_quant=True):
        import comfy.utils
        import folder_paths
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name),
                                                   return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata, dtype=torch.bfloat16)
        vae.throw_exception_if_invalid()
        if fast_rms_rope:
            _use_fast_rms_rope(vae.first_stage_model)
        if fast_convrot_quant:
            _use_fast_convrot(vae)
        return (vae,)


_hip_rotate_quant = None


def _rotate_quant(x2d, group_size, input_act=None, act_weight=None, act_eps=0.0, *, tiled_output=False):
    from . import convrot
    if getattr(_state, "convrot", False) and convrot.supported(x2d, group_size, input_act, act_weight):
        return convrot.rotate_quant_int8(x2d, group_size, input_act, act_weight, act_eps, tiled_output=tiled_output)
    return _hip_rotate_quant(x2d, group_size, input_act, act_weight, act_eps, tiled_output=tiled_output)


def _use_fast_convrot(vae):
    """Route comfy-kitchen's hip ConvRot quantizer to the Triton one, only inside this VAE's decode."""
    global _hip_rotate_quant
    try:
        from comfy_kitchen.backends import hip
    except ImportError:
        return
    if not getattr(hip, "_EXT_AVAILABLE", False) or not hasattr(hip, "_rotate_quant_int8"):
        return
    if _hip_rotate_quant is None:
        _hip_rotate_quant = hip._rotate_quant_int8
        hip._rotate_quant_int8 = _rotate_quant
    decode = vae.decode

    def decode_fast(*args, **kwargs):
        previous = getattr(_state, "convrot", False)
        _state.convrot = True
        try:
            return decode(*args, **kwargs)
        finally:
            _state.convrot = previous

    vae.decode = decode_fast


def _use_fast_rms_rope(model):
    from comfy.ldm.minimax import vae as h3vae
    from . import rmsrope
    _rms_rope = h3vae.Attention.rms_rope

    def rms_rope(self, query, key, rotary_pos_emb):
        if torch.is_grad_enabled():
            return _rms_rope(self, query, key, rotary_pos_emb)
        return rmsrope.rms_rope_split_half_(query, key, rotary_pos_emb, self.qk_norm_scale.to(query.device),
                                            epsilon=self.norm_q.eps, rot_dim=rotary_pos_emb.shape[-3] * 2)

    for m in model.modules():
        if isinstance(m, h3vae.Attention):
            m.rms_rope = types.MethodType(rms_rope, m)


class UnpinnedOffload:
    DESCRIPTION = ("Weights of the MODEL / CLIP passing through are offloaded to RAM without page-locking. Faster model "
                   "swaps when two models do not fit in VRAM together (H3 Image: text encoder <-> DiT on every prompt "
                   "change). Leave the video DiT out: its partial offload during sampling streams from pinned RAM.")
    CATEGORY = "advanced/model"
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"enabled": ("BOOLEAN", {"default": True})},
                "optional": {"model": ("MODEL",), "clip": ("CLIP",)}}

    def apply(self, enabled, model=None, clip=None):
        if model is not None:
            model = model.clone()
            model.model_options[NO_PIN] = enabled
        if clip is not None:
            clip = clip.clone()
            clip.patcher.model_options[NO_PIN] = enabled
        return (model, clip)


NODE_CLASS_MAPPINGS = {"MiniMaxH3TextEncoderBF16": MiniMaxH3TextEncoderBF16,
                       "MiniMaxH3VAELoaderBF16": MiniMaxH3VAELoaderBF16,
                       "H3UnpinnedOffload": UnpinnedOffload}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3TextEncoderBF16": "MiniMax H3 Text Encoder BF16",
                              "MiniMaxH3VAELoaderBF16": "Load VAE (BF16)",
                              "H3UnpinnedOffload": "Unpinned Offload (fast model swap)"}
