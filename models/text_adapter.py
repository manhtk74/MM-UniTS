import torch
import torch.nn.functional as F
from torch import nn


class TextPatchPooler(nn.Module):
    """
    Pool timestep text embeddings into patch-aligned text embeddings.

    Args:
        patch_len: Number of timesteps per UniTS patch.
        stride: Patch stride. MindTS_AD currently uses non-overlapping patches,
            but this module also supports overlapping windows.
        eps: Small value for safe masked averages.

    Inputs:
        text_emb: [B, L, E] timestep-level text embeddings.
        text_mask: optional [B, L] validity mask. If omitted, a timestep is
            valid when its embedding is non-zero. This makes all-zero/null text
            rows safe: they do not contribute to the patch average.

    Outputs:
        pooled: [B, N, E] patch-level text embeddings.
        patch_mask: [B, N] validity mask, 1 when a patch has at least one valid
            text timestep and 0 when all timesteps are null/zero.
    """

    def __init__(self, patch_len, stride, eps=1e-6):
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.patch_len = patch_len
        self.stride = stride
        self.eps = eps

    def forward(self, text_emb, text_mask=None):
        if text_emb is None:
            raise ValueError("text_emb must be a tensor with shape [B, L, E]")
        if text_emb.dim() != 3:
            raise ValueError(
                f"text_emb must have shape [B, L, E], got {tuple(text_emb.shape)}")

        B, L, E = text_emb.shape
        if text_mask is None:
            text_mask = (text_emb.abs().sum(dim=-1) > self.eps).to(text_emb.dtype)
        else:
            if text_mask.shape != (B, L):
                raise ValueError(
                    f"text_mask must have shape {(B, L)}, got {tuple(text_mask.shape)}")
            text_mask = text_mask.to(device=text_emb.device, dtype=text_emb.dtype)

        text_emb = torch.nan_to_num(text_emb)
        padding = self._padding_length(L)
        if padding > 0:
            text_emb = F.pad(text_emb, (0, 0, 0, padding))
            text_mask = F.pad(text_mask, (0, padding))

        text_windows = text_emb.unfold(
            dimension=1, size=self.patch_len, step=self.stride)
        # unfold over dim=1 produces [B, N, E, patch_len].
        text_windows = text_windows.transpose(-1, -2)
        mask_windows = text_mask.unfold(
            dimension=1, size=self.patch_len, step=self.stride)

        weights = mask_windows.unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(self.eps)
        pooled = (text_windows * weights).sum(dim=2) / denom
        patch_mask = (mask_windows.sum(dim=2) > 0).to(text_emb.dtype)
        pooled = pooled * patch_mask.unsqueeze(-1)

        return pooled, patch_mask

    def _padding_length(self, length):
        remainder = length % self.patch_len
        if remainder == 0:
            return 0
        return self.patch_len - remainder


class TextPatchTokenizer(nn.Module):
    """
    Convert timestep text embeddings [B, L, E] into UniTS-dim patch tokens [B, N, D].

    The projection uses bias=False by default so all-zero/null text remains a
    zero token after projection. That keeps missing text inert until later
    modules explicitly decide how to handle it.
    """

    def __init__(self, text_emb_dim, d_model, patch_len, stride, dropout=0.0,
                 bias=False):
        super().__init__()
        self.pooler = TextPatchPooler(patch_len=patch_len, stride=stride)
        self.proj = nn.Linear(text_emb_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_emb, text_mask=None):
        pooled, patch_mask = self.pooler(text_emb, text_mask=text_mask)
        tokens = self.proj(pooled)
        tokens = self.dropout(tokens)
        tokens = tokens * patch_mask.unsqueeze(-1)
        return tokens, patch_mask


class TextCrossAttentionAdapter(nn.Module):
    """
    Cross-attention adapter that injects patch-level text into UniTS sample tokens.

    Inputs:
        H: [B, V, N, D] or [B, N, D] UniTS hidden/sample tokens.
        text_tokens: [B, S, D] patch-level text tokens.
        text_mask: optional [B, S] validity mask. If absent, zero text tokens are
            treated as invalid/null.

    Output:
        H_fused with the same shape as H:
            H + sigmoid(gate) * CrossAttn(LN(H), LN(T), LN(T))

    All-null text rows are handled explicitly: their adapter output is forced to
    zero, so H is returned unchanged and attention does not produce NaNs.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.0, gate_init=-4.0,
                 gate_type="scalar"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if gate_type not in {"scalar", "channel"}:
            raise ValueError("gate_type must be 'scalar' or 'channel'")
        self.d_model = d_model
        self.n_heads = n_heads
        self.gate_type = gate_type
        self.h_norm = nn.LayerNorm(d_model)
        self.t_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        if gate_type == "channel":
            self.gate = nn.Parameter(torch.full((d_model,), float(gate_init)))
        else:
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, H, text_tokens, text_mask=None, return_attn=False):
        if H.dim() not in {3, 4}:
            raise ValueError(
                f"H must have shape [B, N, D] or [B, V, N, D], got {tuple(H.shape)}")
        if text_tokens.dim() != 3:
            raise ValueError(
                f"text_tokens must have shape [B, S, D], got {tuple(text_tokens.shape)}")

        text_tokens = torch.nan_to_num(text_tokens)
        B_text, S, D = text_tokens.shape
        if D != self.d_model:
            raise ValueError(
                f"text token dim {D} does not match adapter d_model {self.d_model}")

        if H.dim() == 4:
            B, V, N, H_D = H.shape
            if B != B_text or H_D != D:
                raise ValueError(
                    f"H shape {tuple(H.shape)} is incompatible with text shape {tuple(text_tokens.shape)}")
            H_flat = H.reshape(B * V, N, D)
            T_flat = text_tokens.repeat_interleave(V, dim=0)
            text_mask_flat = self._prepare_text_mask(
                text_tokens, text_mask).repeat_interleave(V, dim=0)
            restore_shape = (B, V, N, D)
        else:
            B, N, H_D = H.shape
            if B != B_text or H_D != D:
                raise ValueError(
                    f"H shape {tuple(H.shape)} is incompatible with text shape {tuple(text_tokens.shape)}")
            H_flat = H
            T_flat = text_tokens
            text_mask_flat = self._prepare_text_mask(text_tokens, text_mask)
            restore_shape = None

        valid_rows = text_mask_flat.sum(dim=1) > 0
        key_padding_mask = ~(text_mask_flat > 0)
        if (~valid_rows).any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[~valid_rows] = False

        attn_out, attn_weights = self.attn(
            query=self.h_norm(H_flat),
            key=self.t_norm(T_flat),
            value=self.t_norm(T_flat),
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        attn_out = attn_out * valid_rows.to(attn_out.dtype).view(-1, 1, 1)
        fused_flat = H_flat + self.gate_value().view(*self._gate_view_shape()) * attn_out

        if restore_shape is not None:
            fused = fused_flat.reshape(restore_shape)
        else:
            fused = fused_flat

        if return_attn:
            return fused, attn_weights
        return fused

    def gate_value(self):
        return torch.sigmoid(self.gate)

    def _gate_view_shape(self):
        if self.gate_type == "channel":
            return (1, 1, self.d_model)
        return (1, 1, 1)

    def _prepare_text_mask(self, text_tokens, text_mask):
        B, S, _ = text_tokens.shape
        if text_mask is None:
            return (text_tokens.abs().sum(dim=-1) > 0).to(text_tokens.dtype)
        if text_mask.shape != (B, S):
            raise ValueError(
                f"text_mask must have shape {(B, S)}, got {tuple(text_mask.shape)}")
        return text_mask.to(device=text_tokens.device, dtype=text_tokens.dtype)
