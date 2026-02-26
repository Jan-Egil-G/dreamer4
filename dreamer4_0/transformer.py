from typing import Callable


from . import utils as ut


import torch
import torch.nn.functional as F
from torch.nested import nested_tensor

from torch.nn import Module, ModuleList, Embedding, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, full, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange
from torch.utils._pytree import tree_flatten, tree_unflatten

# from hyper_connections import mc_get_init_and_expand_reduce_stream_functions
from einops import einsum, rearrange, repeat, reduce, pack, unpack

from einops.layers.torch import Rearrange, Reduce

from hyper_connections import get_init_and_expand_reduce_stream_functions

from assoc_scan import AssocScan
from collections import namedtuple
from torch_einops_utils import (
    maybe,
    align_dims_left,
    pad_at_dim,
    pad_right_at_dim_to,
    lens_to_mask,
    masked_mean,
    safe_stack,
    safe_cat
)
# try:
#     from torch.nn.attention.flex_attention import flex_attention, create_block_mask
#     if torch.cuda.is_available():
#         flex_attention = torch.compile(flex_attention)
# except ImportError:
#     print("that didnt work")
#     pass

TransformerIntermediates = namedtuple('TransformerIntermediates', ('next_kv_cache', 'normed_time_inputs', 'normed_space_inputs', 'next_rnn_hiddens'))
# rotary embeddings for time
class GRULayer(Module):
    def __init__(
        self,
        dim,
        dim_out
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.gru = nn.GRU(dim, dim_out, batch_first = True)

    def forward(
        self,
        x,
        prev_hiddens = None
    ):
        x = self.norm(x)

        x, hiddens = self.gru(x, prev_hiddens)

        return x, hiddens

def naive_attend(
    q, k, v,
    softclamp_value = None,
    scale = None,
    causal = False,
    causal_block_size = 1,
    mask = None
):

    if not exists(scale):
        scale = q.shape[-1] ** -0.5

    # grouped query attention

    groups = q.shape[1] // k.shape[1]

    q = rearrange(q, 'b (h g) ... -> b h g ...', g = groups)

    # similarity

    sim = einsum(q, k, 'b h g i d, b h j d -> b h g i j')

    # scale and attention

    sim = sim * scale

    # softclamping a la gemma 3

    if exists(softclamp_value):
        sim = softclamp(sim, softclamp_value)

    # masking

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        is_blocked_causal = causal_block_size > 1
        i, j = sim.shape[-2:]

        if is_blocked_causal:
          i = ceil(i / causal_block_size)
          j = ceil(j / causal_block_size)

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)

        if causal_block_size > 1:
            causal_mask = repeat(causal_mask, 'i j -> (i b1) (j b2)', b1 = causal_block_size, b2 = causal_block_size)
            causal_mask = causal_mask[:sim.shape[-2], :sim.shape[-1]]

        sim = sim.masked_fill(causal_mask, mask_value)

    # attend

    attn = sim.softmax(dim = -1)

    # aggregate

    out = einsum(attn, v, 'b h g i j, b h j d -> b h g i d')

    # merge the groups

    return rearrange(out, 'b h g i d -> b (h g) i d')


class Rotary1D(Module):
    def __init__(
        self,
        dim_head,
        theta = 10000.
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(
        self,
        seq_len,
        offset = 0
    ):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype

        t = torch.arange(seq_len, device = device).type(dtype) + offset
        freqs = einsum(t, self.inv_freq, 'i, j -> i j')

        return cat((freqs, freqs), dim = -1)
from einx import add, multiply
class MultiHeadRMSNorm(Module):
    def __init__(
        self,
        dim_head,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** 0.5
        self.gamma = Parameter(torch.zeros(heads, dim_head)) # weight decay friendly

    def forward(
        self,
        x # (b h n d)
    ):
        normed = l2norm(x)
        scale = (self.gamma + 1.) * self.scale
        return multiply('... h n d, h d', normed, scale)


def apply_rotations(
    rotations, # (h n d) | (n d)
    t          # (b h n d)
):

    heads, seq_len, dtype = *t.shape[1:3], t.dtype

    rotations_seq_len = rotations.shape[-2]

    # handle kv caching with rotations

    if rotations_seq_len > seq_len:
        rotations = rotations[-seq_len:]

    # precision

    t = t.float()

    # handle gqa for rotary

    if rotations.ndim == 3 and rotations.shape[0] < heads:
        rotary_heads = rotations.shape[0]

        assert divisible_by(heads, rotary_heads)
        groups = heads // rotary_heads
        rotations = repeat(rotations, 'h ... -> (h g) ...', g = groups)

    x1, x2 = t.chunk(2, dim = -1)
    rotated_half_t = cat((-x2, x1), dim = -1)

    # rotate in the positions

    rotated = t * rotations.cos() + rotated_half_t * rotations.sin()
    return rotated.type(dtype)



def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)
from collections import namedtuple
AttentionIntermediates = namedtuple('AttentionIntermediates', ('next_kv_cache', 'normed_inputs'))



class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        query_heads = None,
        heads = 8,
        pre_rmsnorm = True,
        gate_values = True,
        rmsnorm_query = False, # a paper claims that it is better to just norm only the keys https://openreview.net/forum?id=HkztQWZfl2
        rmsnorm_key = True,
        value_residual = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        # setup grouped query attention

        query_heads = ut.default(query_heads, heads)
        assert query_heads >= heads and ut.divisible_by(query_heads, heads)

        # scaling, splitting and merging of heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        dim_q_inner = dim_head * query_heads
        dim_kv_inner = dim_head * heads

        self.to_q = ut.LinearNoBias(dim, dim_q_inner)
        self.to_k = ut.LinearNoBias(dim, dim_kv_inner)
        self.to_v = ut.LinearNoBias(dim, dim_kv_inner)
        self.to_out = ut.LinearNoBias(dim_q_inner, dim)

        # alphafold gating per head, for attending to nothing

        self.to_gates = None

        if gate_values:
            self.to_gates = Sequential(
                ut.LinearNoBias(dim, query_heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()
            )

        # stability related

        self.q_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = query_heads) if rmsnorm_query else nn.Identity()
        self.k_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = heads) if rmsnorm_key else nn.Identity()

        # value residual

        self.to_learned_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if value_residual else None

    def muon_parameters(self):
        # omit the queries and keys for now given what we learned from kimi 2 paper

        return [
            *self.to_v.parameters(),
            *self.to_out.parameters(),
        ]

    def forward(
        self,
        tokens, # (b n d)
        kv_cache = None,
        return_intermediates = False,
        rotary_pos_emb = None,
        residual_values = None,  # (b n h d)
        attend_fn: Callable | None = None
    ):
        tokens, inverse_packed_batch = ut.pack_one(tokens, '* n d')

        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), self.to_k(tokens), self.to_v(tokens))

        # split heads

        q, k, v = map(self.split_heads, (q, k, v))

        # handle maybe value residual

        if ut.exists(residual_values):
            residual_values = rearrange(residual_values, '... n h d -> (...) h n d')

            assert ut.exists(self.to_learned_value_residual_mix)

            learned_mix = self.to_learned_value_residual_mix(tokens)

            v = v.lerp(residual_values, learned_mix)

        # qk rmsnorm

        q = self.q_heads_rmsnorm(q)
        k = self.k_heads_rmsnorm(k)

        # rotary

        if ut.exists(rotary_pos_emb):
            q = apply_rotations(rotary_pos_emb, q)
            k = apply_rotations(rotary_pos_emb, k)

        # caching

        if ut.exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        # attention

        attend_fn = ut.default(attend_fn, naive_attend)

        out = attend_fn(q, k, v)

        # gate values

        if ut.exists(self.to_gates):
            gates = self.to_gates(tokens)
            out = out * gates

        # merge heads

        out = self.merge_heads(out)

        # combine heads

        out = self.to_out(out)

        out = inverse_packed_batch(out)

        if not return_intermediates:
            return out

        return out, AttentionIntermediates(stack((k, v)), tokens)

# feedforward

class SwiGLUFeedforward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def muon_parameters(self):
        return [
            self.proj_in.weight,
            self.proj_out.weight,
        ]

    def forward(self, x):
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = x * F.gelu(gates)

        return self.proj_out(x)

# axial space time transformer

class AxialSpaceTimeTransformer(Module):
    def __init__(
        self,
        dim,
        depth,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_softclamp_value = 50.,
        time_block_every = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        num_residual_streams = 1,
        num_special_spatial_tokens = 1,
        special_attend_only_itself = False,  # this is set to True for the video tokenizer decoder (latents can only attend to itself while spatial modalities attend to the latents and everything)
        final_norm = True,
        value_residual = True,               # https://arxiv.org/abs/2410.17897 - but with learned mixing from OSS
        rnn_time = False
    ):
        super().__init__()
        assert depth >= time_block_every, f'depth must be at least {time_block_every}'

        # hyper connections

        hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)

        # attention

        self.attn_softclamp_value = attn_softclamp_value

        # attention masking

        self.special_attend_only_itself = special_attend_only_itself

        # time rotary embedding

        self.time_rotary = Rotary1D(attn_dim_head)

        # project initial for value residuals

        self.value_residual = value_residual

        if value_residual:
            dim_inner = attn_dim_head * attn_heads

            self.to_value_residual = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, dim_inner, bias = False),
                Rearrange('... (h d) -> ... h d', h = attn_heads)
            )

        # a gru layer across time

        self.rnn_time = rnn_time
        rnn_layers = []

        # transformer

        layers = []
        is_time = []

        for i in range(depth):
            layer_index = i + 1

            is_time_block = ut.divisible_by(layer_index, time_block_every)
            is_time.append(is_time_block)

            rearrange_to_attend = Rearrange('b t s ... -> b s t ...') if is_time_block else Identity()
            rearrange_from_attend = Rearrange('b s t ... -> b t s ...') if is_time_block else Identity()

            layers.append(ModuleList([
                rearrange_to_attend,
                rearrange_from_attend,
                hyper_conn(branch = Attention(dim = dim, heads = attn_heads, dim_head = attn_dim_head, value_residual = value_residual, **attn_kwargs)),
                hyper_conn(branch = SwiGLUFeedforward(dim = dim, **ff_kwargs))
            ]))

            rnn_layers.append(hyper_conn(branch = GRULayer(dim, dim)) if is_time_block and rnn_time else None)

        self.layers = ModuleList(layers)


        self.rnn_layers = ModuleList(rnn_layers)

        self.is_time = is_time
        # zero-init time attention output projections so they start as identity
        # for is_time_block, (_, _, attn_hyper, _) in zip(self.is_time, self.layers):
        #     if is_time_block:
        #         attn = attn_hyper.branch
        #         nn.init.zeros_(attn.to_out.weight)
        # final norm

        self.final_norm = nn.RMSNorm(dim) if final_norm else nn.Identity()

        # special tokens

        self.num_special_spatial_tokens = num_special_spatial_tokens

    def muon_parameters(self):
        muon_params = []

        for m in self.modules():
            if isinstance(m, (Attention, SwiGLUFeedforward)):
                muon_params.extend(m.muon_parameters())

        return muon_params

    def forward(
        self,
        tokens, # (b t s d)
        cache: TransformerIntermediates | None = None,
        return_intermediates = False

    ): # (b t s d) | (y 2 b h t d)

        batch, time, space_seq_len, _, device = *tokens.shape, tokens.device

        assert tokens.ndim == 4

        # destruct intermediates to cache for attention and rnn respectively

        kv_cache = rnn_prev_hiddens = None

        if ut.exists(cache):
            kv_cache = cache.next_kv_cache
            rnn_prev_hiddens = cache.next_rnn_hiddens

        # attend functions for space and time

        has_kv_cache = ut.exists(kv_cache) 
        use_flex = False#ut.exists(flex_attention) and tokens.is_cuda and not has_kv_cache # KV cache shape breaks flex attention TODO: Fix

        attend_kwargs = dict(use_flex = use_flex, softclamp_value = self.attn_softclamp_value, special_attend_only_itself = self.special_attend_only_itself, device = device)

        space_attend = ut.get_attend_fn(causal = False, seq_len = space_seq_len, k_seq_len = space_seq_len, num_special_tokens = self.num_special_spatial_tokens, **attend_kwargs) # space has an agent token on the right-hand side for reinforcement learning - cannot be attended to by modality

        time_attend = ut.get_attend_fn(causal =False, seq_len = time, k_seq_len = time, **attend_kwargs)

        # prepare cache

        time_attn_kv_caches = []
        rnn_hiddens = []

        if has_kv_cache:
            past_tokens, tokens = tokens[:, :-1], tokens[:, -1:]

            rotary_seq_len = 1
            rotary_pos_offset = past_tokens.shape[1]
        else:
            rotary_seq_len = time
            rotary_pos_offset = 0

        kv_cache = ut.default(kv_cache, (None,))

        iter_kv_cache = iter(kv_cache)

        rnn_prev_hiddens = ut.default(rnn_prev_hiddens, (None,))

        iter_rnn_prev_hiddens = iter(rnn_prev_hiddens)

        # rotary

        rotary_pos_emb = self.time_rotary(rotary_seq_len, offset = rotary_pos_offset)

        # value residual

        residual_values = None

        if self.value_residual:
            residual_values = self.to_value_residual(tokens)

        # normed attention inputs

        normed_time_attn_inputs = []
        normed_space_attn_inputs = []

        # attention

        tokens = self.expand_streams(tokens)

        for (pre_attn_rearrange, post_attn_rearrange, attn, ff), maybe_rnn, layer_is_time in zip(self.layers, self.rnn_layers, self.is_time):

            tokens = pre_attn_rearrange(tokens)

            # maybe rnn for time

            if layer_is_time and ut.exists(maybe_rnn):

                tokens, inverse_pack_batch = ut.pack_one(tokens, '* t d')

                tokens, layer_rnn_hiddens = maybe_rnn(tokens, next(iter_rnn_prev_hiddens, None)) # todo, handle rnn cache

                tokens = inverse_pack_batch(tokens)

                rnn_hiddens.append(layer_rnn_hiddens)

            # when is a axial time attention block, should be causal

            attend_fn = time_attend if layer_is_time else space_attend

            layer_rotary_pos_emb = rotary_pos_emb if layer_is_time else None

            # maybe past kv cache

            maybe_kv_cache = next(iter_kv_cache, None) if layer_is_time else None

            # residual values

            layer_residual_values = maybe(pre_attn_rearrange)(residual_values)

            # attention layer

            tokens, attn_intermediates = attn(
                tokens,
                rotary_pos_emb = layer_rotary_pos_emb,
                attend_fn = attend_fn,
                kv_cache = maybe_kv_cache,
                residual_values = layer_residual_values,
                return_intermediates = True
            )

            tokens = post_attn_rearrange(tokens)

            # feedforward layer

            tokens = ff(tokens)

            # save kv cache if is time layer

            if layer_is_time:
                time_attn_kv_caches.append(attn_intermediates.next_kv_cache)

            # save time attention inputs for decorr

            space_or_time_inputs = normed_time_attn_inputs if layer_is_time else normed_space_attn_inputs

            space_or_time_inputs.append(attn_intermediates.normed_inputs)

        tokens = self.reduce_streams(tokens)

        out = self.final_norm(tokens)

        if has_kv_cache:
            # just concat the past tokens back on for now, todo - clean up the logic
            out = cat((past_tokens, out), dim = 1)

        if not return_intermediates:
            return out


        intermediates = TransformerIntermediates(
            stack(time_attn_kv_caches),
            safe_stack(normed_time_attn_inputs),
            safe_stack(normed_space_attn_inputs),
            safe_stack(rnn_hiddens)
        )

        return out, intermediates
