"""ConvRot int8 activation quantizer (Hadamard-256 rotation + row-wise int8) in one Triton launch.

Replaces comfy-kitchen's hip _rotate_quant_int8 for group size 256 with no input activation or a fused rms_norm, the
two cases the MiniMax H3 video VAE runs at ~110-130 GB/s on gfx1201. Per row: optional RMSNorm (fp32, then the norm
weight), rotation of every 256-wide group by the normalised regular Hadamard matrix, then int8 with a per-row scale
amax / 127. The 256-point Hadamard is H16 (x) H16 (kron of the regular 4x4 one), so a group viewed as a 16x16 matrix X
rotates to H16 X H16 / 16: two 16-wide dots. Output layout: row-major [m, k], or comfy-kitchen's tiled A operand
[ceil(m / 128), k / 128, 128, 128] with rows padded to a multiple of 128.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _convrot_quant_kernel(x_ptr, w_ptr, q_ptr, s_ptr, M, eps,
                          K: tl.constexpr, BM: tl.constexpr, RMS: tl.constexpr, TILED: tl.constexpr):
    G: tl.constexpr = K // 256
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, K)
    rmask = rows < M
    offs = rows.to(tl.int64)[:, None] * K + cols[None, :]
    x = tl.load(x_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
    if RMS:
        inv = tl.rsqrt(tl.sum(x * x, 1) / K + eps)
        w = tl.load(w_ptr + cols).to(tl.float32)
        x = x * inv[:, None] * w[None, :]
    # regular H16 = kron(h4, h4), h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]: entry -1 where the two indices'
    # base-4 digits differ in "sum to 3" fashion, i.e. h4[i, j] = -1 iff i + j == 3
    i = tl.arange(0, 16)
    ii, jj = i[:, None], i[None, :]
    neg = ((ii % 4 + jj % 4) == 3).to(tl.int32) + ((ii // 4 + jj // 4) == 3).to(tl.int32)
    h = tl.where(neg == 1, -0.25, 0.25)  # H16 / 4, so H X H with both factors is the / 16 normalisation
    t = tl.reshape(x, (BM * G * 16, 16))  # rows (row, group, a), cols b
    t = tl.dot(t, h, input_precision="ieee")  # (row, group, a) x d
    t = tl.permute(tl.reshape(t, (BM * G, 16, 16)), (0, 2, 1))  # (row, group), d, a
    t = tl.dot(tl.reshape(t, (BM * G * 16, 16)), h, input_precision="ieee")  # (row, group, d) x c
    y = tl.reshape(tl.permute(tl.reshape(t, (BM * G, 16, 16)), (0, 2, 1)), (BM, K))  # index c * 16 + d
    amax = tl.max(tl.abs(y), 1)
    scale = amax / 127.0
    rcp = tl.where(amax > 0, 127.0 / amax, 0.0)
    q = libdevice.rint(y * rcp[:, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(s_ptr + rows, scale, mask=rmask)
    if TILED:
        r = rows.to(tl.int64)
        qoffs = ((r // 128) * (K * 128) + (r % 128) * 128)[:, None] + ((cols // 128) * (128 * 128) + cols % 128)[None, :]
    else:
        qoffs = offs
    tl.store(q_ptr + qoffs, q, mask=rmask[:, None])


def supported(x2d, group_size, input_act, act_weight):
    k = x2d.shape[-1]
    return (group_size == 256 and x2d.is_cuda and x2d.dim() == 2 and x2d.is_contiguous()
            and x2d.dtype in (torch.bfloat16, torch.float16) and k % 256 == 0 and 256 <= k <= 4096
            and (k & (k - 1)) == 0
            and (input_act in (None, "none") or (input_act == "rms_norm" and act_weight is not None
                                                 and act_weight.numel() == k and act_weight.is_contiguous())))


def rotate_quant_int8(x2d, group_size, input_act=None, act_weight=None, act_eps=0.0, *, tiled_output=False):
    """Same contract as comfy_kitchen.backends.hip._rotate_quant_int8 for the supported() cases: (q, scales)."""
    m, k = x2d.shape
    q_rows = (m + 127) // 128 * 128 if tiled_output else m
    q = torch.empty((q_rows, k), dtype=torch.int8, device=x2d.device)
    scales = torch.empty((m,), dtype=torch.float32, device=x2d.device)
    rms = input_act == "rms_norm"
    bm = 1  # one row per program: best on gfx1201 at K 2048 (428 GB/s vs 262 at 8 rows)
    _convrot_quant_kernel[(triton.cdiv(m, bm),)](
        x2d, act_weight if rms else x2d, q, scales, m, float(act_eps),
        K=k, BM=bm, RMS=rms, TILED=bool(tiled_output), num_warps=4 if k <= 2048 else 8)
    return q, scales
