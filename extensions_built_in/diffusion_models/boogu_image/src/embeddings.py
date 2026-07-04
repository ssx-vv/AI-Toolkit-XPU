# Vendored from the Boogu-Image repository (boogu/models/embeddings.py).
# Original work: Copyright 2024 The HuggingFace Team. Apache-2.0.
#
# Only the pieces the Boogu transformer actually needs are kept here:
# ``TimestepEmbedding`` and ``apply_rotary_emb``.
from typing import Optional, Tuple, Union

import torch
from diffusers.models.activations import get_activation
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight, std=0.02)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Uses real-number operations throughout for broad device compatibility (XPU,
    MPS, CUDA, CPU).  The ``use_real=False`` branch avoids
    ``torch.view_as_complex`` / ``torch.view_as_real`` which are unreliable on
    non-CUDA backends; it performs the equivalent rotation with real arithmetic.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, Boogu and CogView4
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # Lumina / Boogu complex-rotary path, implemented with real-number ops
        # so it works on XPU and other non-CUDA backends where
        # torch.view_as_complex / torch.view_as_real are unreliable.
        # freqs_cis: (S, D) complex -> extract cos/sin via .real/.imag
        freqs = freqs_cis.unsqueeze(2)  # (S, 1, D) complex
        cos = freqs.real  # (S, 1, D) real -- cosine component
        sin = freqs.imag  # (S, 1, D) real -- sine component

        x_half = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x_real = x_half[..., 0]  # (B, H, S, D//2)
        x_imag = x_half[..., 1]  # (B, H, S, D//2)

        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        x_out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
        return x_out.type_as(x)
