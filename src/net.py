from dataclasses import dataclass
import contextlib
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import init
from torch.nn.attention import SDPBackend
import torch.nn.functional as F
from einops import rearrange


def unsqueeze_leading(x, y):
    while x.ndim < y.ndim:
        x = x.unsqueeze(0)
    return x


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    x: multihead features, shape: b h ... d

    freqs_cos,freqs_sin
        frequencies obtained from position coordinates
        shape: (b) (h) ... d//2
        where the batch and head dimensions are optional
    """

    og_dtype = x.dtype
    x = x.float()

    x1, x2 = x.chunk(2, dim=-1)

    # unsqueeze batch dim
    freqs_cos = unsqueeze_leading(freqs_cos, x)
    freqs_sin = unsqueeze_leading(freqs_sin, x)

    x = torch.cat(
        (x1 * freqs_cos - x2 * freqs_sin, x2 * freqs_cos + x1 * freqs_sin), -1
    )

    x = x.to(og_dtype)

    return x


class Rope2DPositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, rope_theta: float = 100.0, head_dim: int = 64):
        super().__init__()

        inv_freq = 1 / rope_theta ** torch.arange(
            0, 1, 4 / head_dim, dtype=torch.float32
        )  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        patch_coords: torch.Tensor,
        dtype=torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.inv_freq.device
        with torch.autocast(device_type=device.type, enabled=False):  # Force float32
            # (b, height * width, 2, head_dim / 4) -> (b, height * width, head_dim / 2) -> (b, height * width, head_dim/2)
            angles = (
                2
                * math.pi
                * patch_coords[..., None]
                * unsqueeze_leading(self.inv_freq, patch_coords)
            )
            angles = rearrange(angles, "... nd df -> ... (nd df)")

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        # unsqueeze head dim
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


@torch.no_grad()
def simple_init_weights_(module, init_std=0.02, torch_rng=None):
    if isinstance(module, nn.Linear):
        init.trunc_normal_(module.weight, std=init_std, generator=torch_rng)
        if module.bias is not None:
            init.zeros_(module.bias)


@torch.no_grad()
def init_block_weights_(
    blocks, init_std=0.02, mup_width_multiplier=1.0, torch_rng=None
):
    """
    `mup_width_multiplier = width / base_width` where base_width is typically 256
    """

    for i, block in enumerate(blocks):
        assert isinstance(block, DiTTransformerBlock)

        ff1_std = init_std / math.sqrt(mup_width_multiplier)
        ff2_std = init_std / math.sqrt(2 * (i + 1) * mup_width_multiplier)

        init.zeros_(block.adaLN_modulation.weight)

        init.trunc_normal_(
            block.attention.proj_q.weight, std=ff1_std, generator=torch_rng
        )
        init.trunc_normal_(
            block.attention.proj_k.weight, std=ff1_std, generator=torch_rng
        )
        init.trunc_normal_(
            block.attention.proj_v.weight, std=ff1_std, generator=torch_rng
        )

        init.trunc_normal_(
            block.attention.proj_g.weight, std=ff1_std, generator=torch_rng
        )

        init.trunc_normal_(
            block.attention.proj_out.weight, std=ff2_std, generator=torch_rng
        )

        init.trunc_normal_(block.mlp.up_proj.weight, std=ff1_std, generator=torch_rng)
        init.trunc_normal_(block.mlp.down_proj.weight, std=ff2_std, generator=torch_rng)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a N-D Tensor of timesteps from 1 to 1000
        embedding_dim (int):
            the dimension of the output.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [... x dim] Tensor of positional embeddings.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim

    emb = torch.exp(exponent)

    # ..., d -> ... d
    timesteps = timesteps.unsqueeze(-1).float()
    emb = timesteps * unsqueeze_leading(emb, timesteps)

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class DerfNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        elementwise_affine: bool = True,
        alpha_init_value=0.5,
        shift_init_value=0.0,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)

    def forward(self, x):
        x = self.alpha * x + self.shift
        if self.elementwise_affine:
            return torch.erf(x) * self.weight + self.bias
        else:
            return torch.erf(x)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}, shift_init_value={self.shift_init_value}"


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 256, use_bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=use_bias)
        self.activation_function = nn.GELU()
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=use_bias)

    def forward(self, x: torch.Tensor):
        x = self.activation_function(self.up_proj(x))
        x = self.down_proj(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        head_dim: int = 64,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        self.proj_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.proj_k = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.proj_v = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        self.proj_g = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        self.q_norm = DerfNorm(head_dim)
        self.k_norm = DerfNorm(head_dim)

        self.proj_out = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )

        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        b, n, _ = features.shape

        # b l d -> b l (h dh)
        q = self.proj_q(features)
        k = self.proj_k(features)
        v = self.proj_v(features)
        g = self.proj_g(features)

        # ... (h dh) -> ... h dh
        q, k, v, g = (
            t.reshape(b, -1, self.num_attention_heads, self.head_dim)
            for t in (q, k, v, g)
        )

        # b s h dh -> b h s dh
        q, k, v, g = (t.transpose(2, 1) for t in (q, k, v, g))

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if rotary_embeds is not None:
            q = apply_rotary_emb(q, *rotary_embeds)
            k = apply_rotary_emb(k, *rotary_embeds)

        if attention_mask is not None:
            if attention_mask.ndim == 3:
                # b l s -> b 1 l s
                attention_mask = attention_mask[:, None, :, :]

        # We need to use math attention during training
        # because we are using forward diff mode
        # when calling jvp
        needs_math_attention = self.training
        attention_context_manager = contextlib.nullcontext()
        if needs_math_attention:
            attention_context_manager = nn.attention.sdpa_kernel(SDPBackend.MATH)

        with attention_context_manager:
            features = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        features = features * F.sigmoid(g)

        # b h l d -> b l h d -> b l (h d)
        features = features.transpose(1, 2).reshape(b, n, -1)
        return self.proj_out(features)


class DiTTransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int = 256, head_dim: int = 64, num_attention_heads: int = 4
    ):
        super().__init__()

        self.attention_pre_norm = DerfNorm(hidden_size)
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=False)

        self.attention = Attention(hidden_size, head_dim, num_attention_heads)

        self.mlp_pre_norm = DerfNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        projected_condition: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if projected_condition is None:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(condition).chunk(6, dim=-1)
            )
        else:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                projected_condition
            )

        residual = hidden_states
        hidden_states = self.attention_pre_norm(hidden_states)
        hidden_states = modulate(hidden_states, shift_attn, scale_attn)
        hidden_states = self.attention(
            hidden_states, attention_mask=attention_mask, rotary_embeds=rotary_embeds
        )
        hidden_states = gate_attn * hidden_states
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_pre_norm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = gate_mlp * hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    freqs: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        use_bias: bool = False,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=use_bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=use_bias),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor):
        # ... -> ... d
        t_freq = get_timestep_embedding(t, embedding_dim=self.frequency_embedding_size)
        t_freq = t_freq.to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


@dataclass
class ViTDenoiserConfig:
    input_size: int = 48
    hidden_size: int = 256
    head_dim: int = 64
    num_attention_heads: int = 4
    num_blocks: int = 12
    should_pin_adaln: bool = True


class ViTDenoiserOutput(NamedTuple):
    prediction: torch.Tensor
    layer_hidden_states: torch.Tensor | None = None


class ViTDenoiser(nn.Module):
    def __init__(self, conf: ViTDenoiserConfig = ViTDenoiserConfig()):
        super().__init__()
        self.conf = conf

        self.proj_in = nn.Linear(
            conf.input_size,
            conf.hidden_size,
            bias=False,
        )

        timestep_frequency_embedding_size = min(conf.hidden_size, 256)
        self.terminal_timestep_embedder = TimestepEmbedder(
            conf.hidden_size, timestep_frequency_embedding_size
        )
        self.timestep_embedder = TimestepEmbedder(
            conf.hidden_size, timestep_frequency_embedding_size
        )

        self.rotary_embeds = Rope2DPositionEmbedding(head_dim=conf.head_dim)

        self.blocks = nn.ModuleList(
            DiTTransformerBlock(
                conf.hidden_size, conf.head_dim, conf.num_attention_heads
            )
            for _ in range(conf.num_blocks)
        )

        self.norm_out = DerfNorm(conf.hidden_size)
        self.proj_out = nn.Linear(conf.hidden_size, conf.input_size, bias=False)

        self.reset_weights_()

    def reset_weights_(
        self, init_std=0.02, mup_width_multiplier: float = 1.0, torch_rng=None
    ):
        self.apply(
            lambda m: simple_init_weights_(m, init_std=init_std, torch_rng=torch_rng)
        )
        init_block_weights_(
            self.blocks,
            init_std=init_std,
            mup_width_multiplier=mup_width_multiplier,
            torch_rng=torch_rng,
        )

        if self.conf.should_pin_adaln:
            adaln_modulation_0 = self.blocks[0].adaLN_modulation
            for block in self.blocks:
                block.adaLN_modulation = adaln_modulation_0

        init.zeros_(self.proj_out.weight)

        return self

    def forward(
        self,
        *,
        patches: torch.Tensor,
        terminal_timesteps: torch.Tensor | None = None,
        timesteps: torch.Tensor,
        patch_coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_layer_indices: list[int] | None = None,
    ):
        """
        --- Tensors ---
        patches: b l d_patch
        timesteps: b/1 l/1
        patch_coords: b/1 l 2
        attention_mask: b/1 l s
        """

        conf = self.conf

        b, s, _ = patches.shape
        device, dtype = patches.device, patches.dtype

        hidden_states = self.proj_in(patches)

        # patch-wise condition (b,s) or batch condition (b,1)
        condition = self.timestep_embedder(timesteps)
        if terminal_timesteps is None:
            terminal_timesteps = torch.zeros_like(timesteps)
        condition = condition + self.terminal_timestep_embedder(terminal_timesteps)
        condition = F.silu(condition)

        projected_condition = None
        if conf.should_pin_adaln:
            block_0 = self.blocks[0]
            adaln_0 = block_0.adaLN_modulation
            assert isinstance(adaln_0, nn.Linear)
            projected_condition = adaln_0(condition).chunk(6, dim=-1)

        rotary_embeds = self.rotary_embeds(patch_coords)

        layer_hidden_states = None
        if return_layer_indices is not None:
            layer_hidden_states = torch.empty(
                len(return_layer_indices),
                b,
                s,
                conf.hidden_size,
                device=device,
                dtype=dtype,
            )

        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                condition=condition,
                projected_condition=projected_condition,
                attention_mask=attention_mask,
                rotary_embeds=rotary_embeds,
            )

            if return_layer_indices is not None:
                assert layer_hidden_states is not None
                if i in return_layer_indices:
                    layer_hidden_states[return_layer_indices.index(i)] = hidden_states

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        return ViTDenoiserOutput(hidden_states, layer_hidden_states)
