# Copyright 2025 DecoVAE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are adapted from the MONAI project
# (https://github.com/Project-MONAI/MONAI), licensed under the Apache
# License 2.0. The `__init__` body mirrors the upstream
# `DiffusionModelUNetMaisi.__init__` verbatim, with the
# meta_layer/use_checkpointing additions described below.
"""
DiffusionModelUNetMaisi with optional meta-feature embedding and gradient checkpointing.

Drop-in replacement for monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi.
Use by setting `_target_` in config JSON to
`patches.diffusion_model_unet_maisi_v2.DiffusionModelUNetMaisiV2`.

Adds two ctor flags:
  - include_meta_input: concatenate a 1D meta-feature embedding to the time embedding
  - use_checkpointing: wrap down/mid/up blocks in torch.utils.checkpoint to trade compute for memory

`__init__` cannot delegate to super().__init__ because the down/mid/up blocks need to be
constructed with the meta-augmented `temb_channels`. The body below mirrors the upstream
__init__ verbatim, with the meta_layer/use_checkpointing additions.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.utils.checkpoint
from torch import nn

from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.blocks import Convolution
from monai.networks.nets.diffusion_model_unet import (
    get_down_block,
    get_mid_block,
    get_timestep_embedding,
    get_up_block,
    zero_module,
)
from monai.utils import ensure_tuple_rep
from monai.utils.type_conversion import convert_to_tensor

__all__ = ["DiffusionModelUNetMaisiV2"]


class DiffusionModelUNetMaisiV2(DiffusionModelUNetMaisi):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_top_region_index_input: bool = False,
        include_bottom_region_index_input: bool = False,
        include_spacing_input: bool = False,
        include_meta_input: bool = False,
        use_checkpointing: bool = False,
        conditioning: dict | None = None,
    ) -> None:
        # Skip DiffusionModelUNetMaisi.__init__ — go directly to nn.Module — because we need
        # `new_time_embed_dim` to include the meta dim before constructing down/mid/up blocks.
        nn.Module.__init__(self)
        self.use_checkpointing = use_checkpointing

        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelUNetMaisi expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNetMaisi expects with_conditioning=True when specifying the cross_attention_dim."
            )
        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")

        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError(
                f"DiffusionModelUNetMaisi expects all num_channels being multiple of norm_num_groups, "
                f"but get num_channels: {num_channels} and norm_num_groups: {norm_num_groups}"
            )
        if len(num_channels) != len(attention_levels):
            raise ValueError(
                f"DiffusionModelUNetMaisi expects num_channels being same size of attention_levels, "
                f"but get num_channels: {len(num_channels)} and attention_levels: {len(attention_levels)}"
            )

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))
        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))
        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )
        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        time_embed_dim = num_channels[0] * 4
        self.time_embed = self._create_embedding_module(num_channels[0], time_embed_dim)

        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        self.include_top_region_index_input = include_top_region_index_input
        self.include_bottom_region_index_input = include_bottom_region_index_input
        self.include_spacing_input = include_spacing_input
        self.include_meta_input = include_meta_input

        new_time_embed_dim = time_embed_dim
        if self.include_top_region_index_input:
            self.top_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_bottom_region_index_input:
            self.bottom_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_spacing_input:
            self.spacing_layer = self._create_embedding_module(3, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_meta_input:
            self.meta_layer = self._create_embedding_module(1, time_embed_dim)
            new_time_embed_dim += time_embed_dim

        # Typed token-set conditioning (pooled foundation model). Alternative to
        # the legacy 1-D meta_layer — like it, the encoded vector is concatenated
        # to the time-embedding. `conditioning.enabled=false` (or absent) → no
        # encoder → fully unconditional / legacy path (MIUA reproducibility).
        self.conditioning = conditioning
        self.use_token_set = bool(conditioning) and bool(conditioning.get("enabled", False))
        if self.use_token_set:
            from patches.token_set_encoder import TokenSetEncoder
            self.token_encoder = TokenSetEncoder(
                attributes=conditioning["attributes"],
                cond_dim=int(conditioning.get("cond_dim", 256)),
                output_dim=time_embed_dim,
                pool=conditioning.get("pool", "mean"),
            )
            new_time_embed_dim += time_embed_dim

        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1
            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )
            self.down_blocks.append(down_block)

        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=new_time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )
            self.up_blocks.append(up_block)

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def _get_input_embeddings(self, emb, top_index, bottom_index, spacing, meta=None):
        if self.include_top_region_index_input:
            emb = torch.cat((emb, self.top_region_index_layer(top_index)), dim=1)
        if self.include_bottom_region_index_input:
            emb = torch.cat((emb, self.bottom_region_index_layer(bottom_index)), dim=1)
        if self.include_spacing_input:
            emb = torch.cat((emb, self.spacing_layer(spacing)), dim=1)
        if self.include_meta_input:
            emb = torch.cat((emb, self.meta_layer(meta)), dim=1)
        if self.use_token_set:
            # `meta` here is a dict {cond_cat, cond_cont, cond_presence}.
            tok = self.token_encoder(meta["cond_cat"], meta["cond_cont"], meta["cond_presence"])
            emb = torch.cat((emb, tok), dim=1)
        return emb

    def _apply_down_blocks(self, h, emb, context, down_block_additional_residuals):
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            if self.use_checkpointing:
                h, res_samples = torch.utils.checkpoint.checkpoint(
                    downsample_block, h, emb, context, use_reentrant=False
                )
            else:
                h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)

        if down_block_additional_residuals is not None:
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples.append(down_block_res_sample)
            down_block_res_samples = new_down_block_res_samples
        return h, down_block_res_samples

    def _apply_up_blocks(self, h, emb, context, down_block_res_samples):
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if self.use_checkpointing:
                h = torch.utils.checkpoint.checkpoint(
                    upsample_block, h, res_samples, emb, context, use_reentrant=False
                )
            else:
                h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        return h

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
        meta_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(
            emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, meta_tensor
        )
        h = self.conv_in(x)
        h, down_res = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        if self.use_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self.middle_block, h, emb, context, use_reentrant=False)
        else:
            h = self.middle_block(h, emb, context)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual
        h = self._apply_up_blocks(h, emb, context, down_res)
        h = self.out(h)
        return convert_to_tensor(h)
