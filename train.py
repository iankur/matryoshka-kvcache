import copy
import warnings
from typing import Any, Optional

import llmfoundry

import torch
import torch.nn as nn
from einops import rearrange
from llmfoundry.layers_registry import attention_classes
from llmfoundry.models.layers.attention import (
    gen_slopes,
    GroupedQueryAttention as OriginalGroupedQueryAttention,
)
from llmfoundry.models.layers.layer_builders import build_fc, build_norm
from llmfoundry.models.mpt import (
    ComposerMPTCausalLM as OriginalComposerMPTCausalLM,
    MPTConfig as OriginalMPTConfig,
    MPTForCausalLM as OriginalMPTForCausalLM,
    MPTModel as OriginalMPTModel,
)
from llmfoundry.models.mpt.modeling_mpt import (
    gen_attention_mask_in_length,
    gen_flash_attn_padding_info,
)
from llmfoundry.models.utils.config_defaults import fc_type_defaults
from transformers.modeling_outputs import BaseModelOutputWithPast


class MPTConfig(OriginalMPTConfig):
    @property
    def allowed_block_overrides(self):
        return {
            "attn_config": {
                "sliding_window_size": None,
                "reuse_kv_layer_idx": None,
                "matryoshka_factor": 1,
                "matryoshka_ascending": True,
            },
            "allow_mismatch": True,
        }


@attention_classes.register_class("grouped_query_attention")
class GroupedQueryAttention(OriginalGroupedQueryAttention):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_n_heads: int,
        attn_impl: str = "flash",
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        qk_gn: bool = False,
        fused_qkv: bool = True,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        norm_eps: float = 1e-05,
        fc_type: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
        reuse_kv_layer_idx: Optional[int] = None,
        attn_logit_softcapping: Optional[float] = None,
        kv_dim: Optional[int] = None,
        **kwargs,
    ):
        self.matryoshka_factor = kwargs.pop("matryoshka_factor", 1)
        self.matryoshka_ascending = kwargs.pop("matryoshka_ascending", True)
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=kv_n_heads,
            attn_impl=attn_impl,
            clip_qkv=clip_qkv,
            qk_ln=qk_ln,
            qk_gn=qk_gn,
            fused_qkv=fused_qkv,
            softmax_scale=softmax_scale,
            attn_pdrop=attn_pdrop,
            norm_type=norm_type,
            norm_eps=norm_eps,
            fc_type=fc_type,
            device=device,
            bias=bias,
            sliding_window_size=sliding_window_size,
            reuse_kv_layer_idx=reuse_kv_layer_idx,
            attn_logit_softcapping=attn_logit_softcapping,
            kv_dim=kv_dim,
        )

        # Usually, fc_type dict should be passed in through MPTBlock's __init__ function.
        if fc_type is None:
            fc_type = copy.deepcopy(fc_type_defaults)
            fc_type["bias"] = kwargs.get("bias", True)
            fc_type["device"] = kwargs.get("device", None)
        fc_type_name = fc_type["name"]

        if self.matryoshka_factor > 1:
            assert self.fused_qkv, "Fused qkv must be enabled"
            self.Wqkv = build_fc(
                name=fc_type_name,
                in_features=self.d_model,
                out_features=self.d_model
                + 2 * self.kv_n_heads * self.head_dim // self.matryoshka_factor,
                fc_kwargs=fc_type,
            )
            # for param init fn; enables shape based init of fused layers
            fuse_splits = [i * self.head_dim for i in range(1, self.n_heads + 1)] + [
                self.n_heads * self.head_dim
                + i * self.head_dim // self.matryoshka_factor
                for i in range(1, 2 * self.kv_n_heads)
            ]
            self.Wqkv._fused = (0, fuse_splits)

            if self.qk_ln or self.qk_gn:
                norm_size = self.head_dim if qk_gn else d_model
                self.q_ln = build_norm(
                    name=norm_type.lower(),
                    normalized_shape=norm_size,
                    eps=norm_eps,
                    device=device,
                )
                if self.reuse_kv_layer_idx is None:
                    if qk_ln:
                        norm_size = self.head_dim * kv_n_heads // self.matryoshka_factor
                    self.k_ln = build_norm(
                        name=norm_type.lower(),
                        normalized_shape=norm_size,
                        eps=norm_eps,
                        device=device,
                    )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb_w_meta_info: Optional[dict] = None,
        is_causal: bool = True,
        needs_weights: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
        prev_layer_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[tuple[torch.Tensor, torch.Tensor]],
    ]:
        extra_kwargs = {}
        if prev_layer_key_value is not None:
            extra_kwargs["prev_layer_key_value"] = prev_layer_key_value
        query, key, value = self.get_qkv(
            x=x,
            key_value_states=key_value_states,
            **extra_kwargs,
        )

        if rotary_emb_w_meta_info is not None:
            query, key, value = self._apply_rotary_embeddings(
                rotary_emb_w_meta_info,
                query,
                key,
                value,
            )

        if self.matryoshka_factor > 1:
            key, value = self.matryoshka_key_value(key, value, prev_layer_key_value)

        extra_attn_kwargs = self.get_implementation_specific_args(
            attention_mask,
            alibi_slopes,
            flash_attn_padding_info,
        )

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            n_heads=self.n_heads,
            kv_n_heads=self.kv_n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            attn_logit_softcapping=self.attn_logit_softcapping,
            sliding_window_size=self.sliding_window_size,
            **extra_attn_kwargs,
        )

        return self.out_proj(context), attn_weights, past_key_value

    def get_qkv(
        self,
        x: torch.Tensor,
        prev_layer_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes and returns the query, key, and value tensors.

        Args:
            x (torch.Tensor): The input query tensor.
            prev_layer_key_value  (Optional[Tuple[torch.Tensor, torch.Tensor]]): The key value of the previous layer.
            key_value_states (Optional[torch.Tensor]): The input tensor for keys and values.

        Returns:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
        """
        if self.reuse_kv_layer_idx is not None:
            if prev_layer_key_value is None:
                raise ValueError(
                    "prev_layer_key_value is None, cannot reuse_prev_layer_kv.",
                )
            key, value = prev_layer_key_value
            if self.attn_impl == "torch":
                key = rearrange(key, "b h d s -> b s (h d)")
                value = rearrange(value, "b h s d -> b s (h d)")

            query = self.Wq(x)
            if self.clip_qkv:
                query = query.clamp(min=-self.clip_qkv, max=self.clip_qkv)

            if self.qk_ln or self.qk_gn:
                # Applying layernorm to qk
                q_shape = query.shape
                if self.qk_gn:
                    b, s = query.shape[:2]
                    query = query.view(b, s, self.n_heads, -1)
                dtype = query.dtype
                query = self.q_ln(query).to(dtype).view(q_shape)
            return query, key, value

        if self.fused_qkv:
            if key_value_states is not None:
                raise ValueError(
                    "Cannot use separate hidden and key_value states when fused_qkv = True.",
                )
            qkv = self.Wqkv(x)

            if self.clip_qkv:
                qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

            query, key, value = qkv.split(
                [
                    self.d_model,
                    self.kv_n_heads * self.head_dim // self.matryoshka_factor,
                    self.kv_n_heads * self.head_dim // self.matryoshka_factor,
                ],
                dim=2,
            )

        if self.qk_ln or self.qk_gn:
            # Applying layernorm to qk
            q_shape, k_shape = query.shape, key.shape
            if self.qk_gn:
                b, s = query.shape[:2]
                query = query.view(b, s, self.n_heads, -1)
                key = key.view(b, s, self.kv_n_heads, -1)
            dtype = query.dtype
            query = self.q_ln(query).to(dtype).view(q_shape)
            key = self.k_ln(key).to(dtype).view(k_shape)

        # hack: repeat key and value for subsequent positional encoding
        if self.matryoshka_factor > 1:
            b, s = key.shape[:2]
            key = key.view(b, s, self.kv_n_heads, -1).repeat(
                1, 1, 1, self.matryoshka_factor
            )
            key = key.reshape(b, s, -1)
            value = value.view(b, s, self.kv_n_heads, -1).repeat(
                1, 1, 1, self.matryoshka_factor
            )
            value = value.reshape(b, s, -1)

        return query, key, value

    def matryoshka_key_value(self, key, value, prev_layer_key_value):
        """
        Overwrite first (last) few dims of previous key and value with
        current key and value if matryoshka_ascending is true (false).
        """
        assert len(key.shape) == 3
        bsz, seqlen = key.shape[:2]
        matryoshka_dim = self.head_dim // self.matryoshka_factor

        if matryoshka_dim == self.head_dim:
            return key, value

        prev_key, prev_value = prev_layer_key_value
        prev_key = prev_key.reshape(-1, self.head_dim)
        prev_value = prev_value.reshape(-1, self.head_dim)

        key = key.reshape(-1, self.head_dim)
        if self.matryoshka_ascending:
            prev_key[:, :matryoshka_dim] = key[:, :matryoshka_dim]
        else:
            prev_key[:, -matryoshka_dim:] = key[:, -matryoshka_dim:]

        value = value.reshape(-1, self.head_dim)
        if self.matryoshka_ascending:
            prev_value[:, :matryoshka_dim] = value[:, :matryoshka_dim]
        else:
            prev_value[:, -matryoshka_dim:] = value[:, -matryoshka_dim:]

        key = prev_key.reshape(bsz, seqlen, -1)
        value = prev_value.reshape(bsz, seqlen, -1)
        return key, value


@attention_classes.register_class("multihead_attention")
class MultiheadAttention(GroupedQueryAttention):
    """Multi-head self attention.

    Using torch attention implementation enables user to also use additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = "flash",
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        qk_gn: bool = False,
        fused_qkv: bool = True,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        norm_eps: float = 1e-05,
        fc_type: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
        reuse_kv_layer_idx: Optional[int] = None,
        attn_logit_softcapping: Optional[float] = None,
        kv_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=n_heads,  # for MHA, same # heads as kv groups
            attn_impl=attn_impl,
            clip_qkv=clip_qkv,
            qk_ln=qk_ln,
            qk_gn=qk_gn,
            fused_qkv=fused_qkv,
            softmax_scale=softmax_scale,
            attn_pdrop=attn_pdrop,
            norm_type=norm_type,
            norm_eps=norm_eps,
            fc_type=fc_type,
            device=device,
            bias=bias,
            sliding_window_size=sliding_window_size,
            reuse_kv_layer_idx=reuse_kv_layer_idx,
            attn_logit_softcapping=attn_logit_softcapping,
            kv_dim=kv_dim,
            **kwargs,
        )


@attention_classes.register_class("multiquery_attention")
class MultiQueryAttention(GroupedQueryAttention):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = "flash",
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        qk_gn: bool = False,
        fused_qkv: bool = True,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        norm_eps: float = 1e-05,
        fc_type: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
        reuse_kv_layer_idx: Optional[int] = None,
        attn_logit_softcapping: Optional[float] = None,
        kv_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=1,  # for MQA, 1 head
            attn_impl=attn_impl,
            clip_qkv=clip_qkv,
            qk_ln=qk_ln,
            qk_gn=qk_gn,
            fused_qkv=fused_qkv,
            softmax_scale=softmax_scale,
            attn_pdrop=attn_pdrop,
            norm_type=norm_type,
            norm_eps=norm_eps,
            fc_type=fc_type,
            device=device,
            bias=bias,
            sliding_window_size=sliding_window_size,
            reuse_kv_layer_idx=reuse_kv_layer_idx,
            attn_logit_softcapping=attn_logit_softcapping,
            kv_dim=kv_dim,
            **kwargs,
        )


class MPTModel(OriginalMPTModel):
    config_class = MPTConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for b_idx, block in enumerate(self.blocks):
            attn_block = (
                block.norm_attn_norm.attn
                if self.blocks_fuse_norm_attn_norm
                else block.attn
            )
            if attn_block.matryoshka_factor > 1:
                self.kv_cache_layers.add(b_idx - 1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if attention_mask is not None:
            attention_mask = attention_mask.bool()  # type: ignore

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MPTModel
        if not return_dict:
            raise NotImplementedError(
                "return_dict False is not implemented yet for MPT",
            )
        if output_attentions:
            if self.attn_impl != "torch":
                raise NotImplementedError(
                    "output_attentions is not implemented for MPT when using attn_impl `flash`.",
                )

        if (
            self.training
            and attention_mask is not None
            and attention_mask[:, 0].sum() != attention_mask.shape[0]
        ):
            raise NotImplementedError(
                "MPT does not support training with left padding.",
            )

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    "sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True "
                    + "and the model is in train mode.",
                )
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                warnings.warn(
                    "MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. "
                    + "This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.",
                )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds.",
            )
        elif input_ids is not None:
            bsz = input_ids.size(0)
            x = self.wte(input_ids)
            input_device = input_ids.device
        elif inputs_embeds is not None:
            bsz = inputs_embeds.size(0)
            x = inputs_embeds
            input_device = inputs_embeds.device
        else:
            raise ValueError("You must specify input_ids or inputs_embeds")

        S = self.get_sequence_length(x)  # noqa: N806

        assert (
            S <= self.config.max_seq_len
        ), f"Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}"

        rotary_emb_w_meta_info = None

        past_position = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(
                    "past_key_values must provide a past_key_value for each attention "
                    + f"layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).",
                )
            # For attn_impl: flash, the past key tensor spec is (batch, seq, dim).
            # For attn_impl: torch, the past key tensor spec is (batch, heads, head_dim, seq).
            # Here we shift position embedding using the `seq` dim of the past key
            past_position = past_key_values[0][0].size(1)
            if self.attn_impl == "torch":
                past_position = past_key_values[0][0].size(3)

        if self.learned_pos_emb or self.rope:
            if self.learned_pos_emb and (S + past_position > self.config.max_seq_len):
                raise ValueError(
                    f"Cannot forward input with past sequence length {past_position} and current sequence length "
                    + f"{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.",
                )

            if self.learned_pos_emb or (self.rope and self.rope_impl == "hf"):
                if position_ids is None:
                    pos = torch.arange(
                        past_position,
                        S + past_position,
                        dtype=torch.long,
                        device=input_device,
                    ).unsqueeze(0)
                else:
                    pos = position_ids

                if attention_mask is not None:
                    # adjust the position indices to account for padding tokens
                    pos = torch.clamp(
                        pos
                        - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[
                            :, past_position:
                        ],
                        min=0,
                    )
                if self.learned_pos_emb:
                    x = x + self.wpe(pos)
                elif self.rope and self.rope_impl == "hf":
                    rotary_emb_w_meta_info = {
                        "impl": self.rope_impl,
                        "rotary_emb": self.rotary_embedding,
                        "offset_info": pos,
                        "seq_len": S + past_position,
                    }
            elif self.rope and self.rope_impl == "dail":
                rotary_emb_w_meta_info = {
                    "impl": self.rope_impl,
                    "rotary_emb": self.rotary_embedding,
                    "offset_info": past_position,
                    "seq_len": S + past_position,
                }

        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction)
            )
            assert isinstance(self.emb_drop, nn.Module)  # pyright
            x = self.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=torch.float32,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
        )
        attention_mask_in_length = gen_attention_mask_in_length(
            sequence_id=sequence_id,
            S=S,
            attn_uses_sequence_id=self.attn_uses_sequence_id,
            attn_impl=self.attn_impl,
            attention_mask=attention_mask,
        )

        alibi_slopes = (
            None  # alibi_slopes will only be used by flash attention for ALiBi
        )
        if self.alibi and self.attn_impl == "flash":
            alibi_slopes = gen_slopes(
                n_heads=self.config.n_heads,
                alibi_bias_max=self.alibi_bias_max,
                device=x.device,
                return_1d=True,
            )

        # initialize the past key values cache if it should be used
        presents = () if use_cache else None
        if (use_cache or len(self.kv_cache_layers) > 0) and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        flash_attn_padding_info = {}
        if self.attn_impl == "flash":
            flash_attn_padding_info = gen_flash_attn_padding_info(
                bsz,
                S,
                past_position,
                x.device,
                attention_mask_in_length,
                attention_mask,
            )

        layer_kv_cache_dict = {}
        for b_idx, block in enumerate(self.blocks):
            attn_block = (
                block.norm_attn_norm.attn
                if self.blocks_fuse_norm_attn_norm
                else block.attn
            )
            if attn_block.reuse_kv_layer_idx is not None:
                if attn_block.reuse_kv_layer_idx not in layer_kv_cache_dict:
                    raise KeyError(
                        f"kv cache for layer {block.reuse_kv_layer_idx} not found in {layer_kv_cache_dict=}.",
                    )
                prev_layer_key_value = layer_kv_cache_dict[
                    attn_block.reuse_kv_layer_idx
                ]
            elif attn_block.matryoshka_factor > 1:
                prev_layer_key_value = layer_kv_cache_dict.get(b_idx - 1, None)
            else:
                prev_layer_key_value = None
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = (
                past_key_values[b_idx] if past_key_values is not None else None
            )
            extra_kwargs = {}
            if prev_layer_key_value is not None:
                extra_kwargs["prev_layer_key_value"] = prev_layer_key_value
            x, attn_weights, present = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=self.is_causal,
                output_attentions=bool(output_attentions),
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
                **extra_kwargs,
            )
            if presents is not None:
                presents += (present,)
            if b_idx in self.kv_cache_layers:
                layer_kv_cache_dict[b_idx] = [
                    present[0][:, past_position:],
                    present[1][:, past_position:],
                ]

            if output_attentions:
                assert all_self_attns is not None  # pyright
                all_self_attns = all_self_attns + (attn_weights,)

        x = self.norm_f(x)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            assert all_hidden_states is not None  # pyright
            all_hidden_states = all_hidden_states + (x,)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MPTForCausalLM(OriginalMPTForCausalLM):
    @property
    def backbone_model_class(self) -> type[OriginalMPTModel]:
        return MPTModel


class ComposerMPTCausalLM(OriginalComposerMPTCausalLM):
    @property
    def model_class(self) -> type[OriginalMPTForCausalLM]:
        return MPTForCausalLM

    @property
    def config_class(self) -> type[OriginalMPTConfig]:
        return MPTConfig


llmfoundry.registry.models.register("mpt_causal_lm", func=ComposerMPTCausalLM)


if __name__ == "__main__":
    import sys

    from llmfoundry.command_utils import train_from_yaml

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    train_from_yaml(yaml_path, args_list)
