"""In-place per-head RMSNorm + partial split-half RoPE for q and k in one Triton launch.

Drop-in for comfy_kitchen.rms_rope_split_half_ on the MiniMax H3 video VAE's layout: q and k are strided chunks of
the fused qkv output ([B, L, H, 3 * D] -> [B, L, H, D] views), freqs_cis is [B, L, 1, rot_dim / 2, 2, 2], one 1-D
scale of length D shared by every head. Each program normalises and rotates a tile of (token, head) rows of both q
and k, computing in fp32 and rounding once. Anything outside that layout goes to comfy-kitchen.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rows(x_ptr, n, nmask, off_row, f_row, f_ptr, f_s1, f_s2, f_s3, w0, w1, w2, eps,
          D: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, P: tl.constexpr, PP: tl.constexpr):
    # first rotated half [0, R), second rotated half [R, 2R), pass-through [2R, D)
    i = tl.arange(0, RP)
    p = tl.arange(0, PP)
    mi = nmask[:, None] & (i < R)[None, :]
    mp = nmask[:, None] & (p < P)[None, :]
    base = x_ptr + off_row[:, None]
    x1 = tl.load(base + i[None, :], mask=mi, other=0.0).to(tl.float32)
    x2 = tl.load(base + R + i[None, :], mask=mi, other=0.0).to(tl.float32)
    x3 = tl.load(base + 2 * R + p[None, :], mask=mp, other=0.0).to(tl.float32)
    ss = tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1) + tl.sum(x3 * x3, 1)
    inv = tl.rsqrt(ss / D + eps)[:, None]
    x1 = x1 * inv * w0[None, :]
    x2 = x2 * inv * w1[None, :]
    x3 = x3 * inv * w2[None, :]
    fb = f_ptr + f_row[:, None] + i[None, :] * f_s1
    f00 = tl.load(fb, mask=mi, other=0.0).to(tl.float32)
    f01 = tl.load(fb + f_s3, mask=mi, other=0.0).to(tl.float32)
    f10 = tl.load(fb + f_s2, mask=mi, other=0.0).to(tl.float32)
    f11 = tl.load(fb + f_s2 + f_s3, mask=mi, other=0.0).to(tl.float32)
    o1 = f00 * x1 + f01 * x2
    o2 = f10 * x1 + f11 * x2
    ty = x_ptr.dtype.element_ty
    tl.store(base + i[None, :], o1.to(ty), mask=mi)
    tl.store(base + R + i[None, :], o2.to(ty), mask=mi)
    tl.store(base + 2 * R + p[None, :], x3.to(ty), mask=mp)


@triton.jit
def _rms_rope_kernel(q_ptr, k_ptr, f_ptr, w_ptr, N, H, L, q_sb, q_sl, q_sh, k_sb, k_sl, k_sh,
                     f_sb, f_sl, f_s1, f_s2, f_s3, eps,
                     D: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, P: tl.constexpr, PP: tl.constexpr,
                     BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    nmask = n < N
    h = n % H
    t = n // H
    l = t % L
    b = t // L
    i = tl.arange(0, RP)
    p = tl.arange(0, PP)
    w0 = tl.load(w_ptr + i, mask=i < R, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + R + i, mask=i < R, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + 2 * R + p, mask=p < P, other=0.0).to(tl.float32)
    f_row = b.to(tl.int64) * f_sb + l.to(tl.int64) * f_sl
    q_row = b.to(tl.int64) * q_sb + l.to(tl.int64) * q_sl + h.to(tl.int64) * q_sh
    k_row = b.to(tl.int64) * k_sb + l.to(tl.int64) * k_sl + h.to(tl.int64) * k_sh
    _rows(q_ptr, n, nmask, q_row, f_row, f_ptr, f_s1, f_s2, f_s3, w0, w1, w2, eps, D, R, RP, P, PP)
    _rows(k_ptr, n, nmask, k_row, f_row, f_ptr, f_s1, f_s2, f_s3, w0, w1, w2, eps, D, R, RP, P, PP)


def _supported(q, k, freqs, q_scale, k_scale, rot_dim):
    if not (q.is_cuda and q.dim() == 4 and k is not None and k.shape == q.shape and k.dtype == q.dtype):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16) or q.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    B, L, H, D = q.shape
    R = rot_dim // 2
    if not (0 < rot_dim <= D and rot_dim % 2 == 0 and D - rot_dim >= 1):
        return False
    if freqs.dim() != 6 or tuple(freqs.shape) != (freqs.shape[0], L, 1, R, 2, 2) or freqs.shape[0] not in (1, B):
        return False
    if k_scale is not None and k_scale is not q_scale and not torch.equal(k_scale, q_scale):
        return False
    return q_scale.dim() == 1 and q_scale.numel() == D and q_scale.device == q.device and freqs.device == q.device


def rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    """Same contract as comfy_kitchen.rms_rope_split_half_ (in place, returns (q, k))."""
    if not _supported(q, k, freqs_cis, q_scale, k_scale, rot_dim):
        import comfy.quant_ops
        return comfy.quant_ops.ck.rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale, epsilon=epsilon,
                                                        rot_dim=rot_dim)
    B, L, H, D = q.shape
    R = rot_dim // 2
    P = D - 2 * R
    N = B * L * H
    BLOCK = 64
    f_sb = freqs_cis.stride(0) if freqs_cis.shape[0] > 1 else 0
    w = q_scale if q_scale.is_contiguous() else q_scale.contiguous()
    _rms_rope_kernel[(triton.cdiv(N, BLOCK),)](
        q, k, freqs_cis, w, N, H, L, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        f_sb, freqs_cis.stride(1), freqs_cis.stride(3), freqs_cis.stride(4), freqs_cis.stride(5), float(epsilon),
        D=D, R=R, RP=triton.next_power_of_2(R), P=P, PP=triton.next_power_of_2(P), BLOCK=BLOCK, num_warps=4)
    return q, k
