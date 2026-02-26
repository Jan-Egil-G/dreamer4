from __future__ import annotations

import math
from math import ceil, log2
from random import random
import random as random_module
from contextlib import nullcontext
from collections import namedtuple
from functools import partial, wraps
from dataclasses import dataclass, asdict
import dreamer4_0.utils as ut
import torch
import torch.nn.functional as F
from torch.nested import nested_tensor
from torch.distributions import Normal, kl
from torch.nn import Module, ModuleList, Embedding, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, full, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange
from torch.utils._pytree import tree_flatten, tree_unflatten
from . import transformer as trafo
import torchvision
from torchvision.models import VGG16_Weights

from torch.optim import Optimizer
from adam_atan2_pytorch import MuonAdamAtan2

from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import create_mlp

from hyper_connections import get_init_and_expand_reduce_stream_functions

from assoc_scan import AssocScan

import logging

logger = logging.getLogger(__name__)


logger.debug("Dreamer processing")

# b - batch
# n - sequence
# h - attention heads
# d - feature dimension
# f - frequencies (rotary)
# l - logit / predicted bins
# y - layers of transformer
# p - positions (3 for spacetime in this work)
# t - time
# na - action dimension (number of discrete and continuous actions)
# g - groups of query heads to key heads (gqa)
# vc - video channels
# vh, vw - video height and width
# mtp - multi token prediction length
# v - video viewpoints
class Print(nn.Module):
    def __init__(self, message=""):
        super().__init__()
        self.message = message
    
    def forward(self, x):
        print(f"{self.message} shape: {x.shape}")
        return x
import einx
from einx import add, multiply
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# flex attention - but will make sure it works if it is not available
# may also end up crafting own custom flash attention kernel for this work

flex_attention = None



LinearNoBias = partial(Linear, bias = False)

TokenizerLosses = namedtuple('TokenizerLosses', ('recon', 'lpips'))

WorldModelLosses = namedtuple('WorldModelLosses', ('flow', 'rewards', 'discrete_actions', 'continuous_actions'))
import os

import torch
import torchvision

# attention

# video tokenizer

# dynamics model, axial space-time transformer

class DynamicsWorldModel(Module):
    def __init__(
        self,
        dim,
        dim_latent,

        max_steps = 64,                # K_max in paper
        num_register_tokens = 4,       # they claim register tokens led to better temporal consistency
        num_spatial_tokens = 2,        # latents projected to greater number of spatial tokens
        num_latent_tokens = None,
        num_agents = 1,
        num_tasks = 1,
        num_video_views = 1,
        dim_proprio = None,
        reward_encoder_kwargs: dict = dict(),
        depth = 4,
        pred_orig_latent = True,   # directly predicting the original x0 data yield better results, rather than velocity (x-space vs v-space)
        time_block_every = 4,      # every 4th block is time
        attn_kwargs: dict = dict(
            heads = 8,
        ),
        transformer_kwargs: dict = dict(),
        attn_dim_head = 32,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        loss_weight_fn: Callable = ut.ramp_weight,
        num_future_predictions = 8,                 # they do multi-token prediction of 8 steps forward
        prob_no_shortcut_train = None,              # probability of no shortcut training, defaults to 1 / num_step_sizes
        add_reward_embed_to_agent_token = False,
        add_reward_embed_dropout = 0.1,
        num_discrete_actions: int | tuple[int, ...] = (2,) * 17,
        multi_token_pred_len = 8,
        value_head_mlp_depth = 3,
        policy_head_mlp_depth = 3,
        latent_flow_loss_weight = 1.,
        reward_loss_weight: float | list[float] = 1.,
        discrete_action_loss_weight: float | list[float] = 1.,
        continuous_action_loss_weight: float | list[float] = 1.,
        num_latent_genes = 0,                       # for carrying out evolution within the dreams https://web3.arxiv.org/abs/2503.19037
        num_residual_streams = 1,
        keep_reward_ema_stats = False,
        reward_ema_decay = 0.998,
        reward_quantile_filter = (0.05, 0.95),
        gae_discount_factor = 0.997,
        gae_lambda = 0.95,
        ppo_eps_clip = 0.2,
        pmpo_pos_to_neg_weight = 0.5, # pos and neg equal weight
        pmpo_reverse_kl = True,
        pmpo_kl_div_loss_weight = .3,
        value_clip = 0.4,
        policy_entropy_weight = .01,
        gae_use_accelerated = False
    ):
        super().__init__()

        # can accept raw video if tokenizer is passed in


        # if ut.exists(video_tokenizer):
        #     num_latent_tokens = ut.default(num_latent_tokens, video_tokenizer.num_latent_tokens)
        #     assert video_tokenizer.num_latent_tokens == num_latent_tokens, f'`num_latent_tokens` must be the same for the tokenizer and dynamics model'

        assert ut.exists(num_latent_tokens), '`num_latent_tokens` must be set'

        # spatial

        self.num_latent_tokens = num_latent_tokens
        self.dim_latent = dim_latent
        self.dim = dim
        self.latent_shape = (num_latent_tokens, dim_latent)
        expand_factor = 1#only support equal amount for now

        self.latents_to_spatial_tokens = Sequential(
            Linear(self.dim_latent, self.dim * expand_factor),
            Rearrange('... (s d) -> ... s d', s = expand_factor)
        ).cuda()

        if num_spatial_tokens >= num_latent_tokens:
            assert ut.divisible_by(num_spatial_tokens, num_latent_tokens)

            expand_factor = num_spatial_tokens // num_latent_tokens

            self.latents_to_spatial_tokens = Sequential(
                Linear(dim_latent, dim * expand_factor),
                Rearrange('... (s d) -> ... s d', s = expand_factor)
            ).cuda()

            self.to_latent_pred = Sequential(
                Reduce('b t n s d -> b t n d', 'mean'),
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent)
            ).cuda()

        else:
            assert ut.divisible_by(num_latent_tokens, num_spatial_tokens)
            latent_tokens_to_space = num_latent_tokens // num_spatial_tokens

            self.latents_to_spatial_tokens = Sequential(
                Rearrange('... n d -> ... (n d)'),
                Linear(num_latent_tokens * dim_latent, dim * num_spatial_tokens),
                Rearrange('... (s d) -> ... s d', s = num_spatial_tokens)
            ).cuda()

            self.to_latent_pred = Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent * latent_tokens_to_space),
                Rearrange('b t s (n d) -> b t (s n) d', n = latent_tokens_to_space)
            )

        # number of video views, for robotics, which could have third person + wrist camera at least

        assert num_video_views >= 1
        self.video_has_multi_view = num_video_views > 1

        self.num_video_views = num_video_views





        # register tokens

        self.num_register_tokens = num_register_tokens
        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2).cuda()

        # signal and step sizes

        assert ut.divisible_by(dim, 2)
        dim_half = dim // 2

        assert ut.is_power_two(max_steps), '`max_steps` must be a power of 2'
        self.max_steps = max_steps
        self.num_step_sizes_log2 = int(log2(max_steps))

        self.signal_levels_embed = nn.Embedding(max_steps, dim_half).cuda()
        self.step_size_embed = nn.Embedding(self.num_step_sizes_log2, dim_half).cuda() # power of 2, so 1/1, 1/2, 1/4, 1/8 ... 1/Kmax

        self.prob_no_shortcut_train = ut.default(prob_no_shortcut_train, self.num_step_sizes_log2 ** -1.)

        # loss related

        self.pred_orig_latent = pred_orig_latent # x-space or v-space
        self.loss_weight_fn = loss_weight_fn

        # reinforcement related


        # they sum all the actions into a single token

        self.num_agents = num_agents

        self.agent_learned_embed = Parameter(randn(self.num_agents, dim) * 1e-2)
        self.action_learned_embed = Parameter(randn(self.num_agents, dim) * 1e-2)

        self.reward_learned_embed = Parameter(randn(self.num_agents, dim) * 1e-2)

        self.num_tasks = num_tasks
        self.task_embed = nn.Embedding(num_tasks, dim)

        # learned set of latent genes

        self.agent_has_genes = num_latent_genes > 0
        self.num_latent_genes = num_latent_genes
        self.latent_genes = Parameter(randn(num_latent_genes, dim) * 1e-2)

        # policy head

        self.policy_head_online = create_mlp(
            dim_in = dim,
            dim = dim * 4,
            dim_out = dim * 4,
            depth = policy_head_mlp_depth
        ).cuda()

        self.policy_head_lagging = create_mlp(
            dim_in = dim,
            dim = dim * 4,
            dim_out = dim * 4,
            depth = policy_head_mlp_depth
        ).cuda()


        # action embedder

        self.action_embedder = ut.ActionEmbedder(
            dim = dim,
            num_discrete_actions = num_discrete_actions,

            can_unembed = True,
            unembed_dim = dim * 4,
            num_unembed_preds = multi_token_pred_len,
            squeeze_unembed_preds = False
        ).cuda()

        # multi token prediction length

        self.multi_token_pred_len = multi_token_pred_len

        # each agent token will have the reward embedding of the previous time step - but could eventually just give reward its own token

        self.add_reward_embed_to_agent_token = add_reward_embed_to_agent_token
        self.add_reward_embed_dropout = add_reward_embed_dropout

        self.reward_encoder = ut.SymExpTwoHot(
            **reward_encoder_kwargs,
            dim_embed = dim,
            learned_embedding = add_reward_embed_to_agent_token
        )

        to_reward_pred = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, self.reward_encoder.num_bins)
        )

        self.to_reward_pred = Ensemble(
            to_reward_pred,
            multi_token_pred_len
        ).cuda()


        # value head

        self.value_head = create_mlp(
            dim_in = dim,
            dim = dim * 4,
            dim_out = self.reward_encoder.num_bins,
            depth = value_head_mlp_depth,
        ).cuda()


        # efficient axial space / time transformer

        self.transformer = trafo.AxialSpaceTimeTransformer(
            dim = dim,
            depth = depth,
            attn_dim_head = attn_dim_head,
            attn_softclamp_value = attn_softclamp_value,
            attn_kwargs = attn_kwargs,
            ff_kwargs = ff_kwargs,
            num_residual_streams = num_residual_streams,
            num_special_spatial_tokens = num_agents,
            time_block_every = time_block_every,
            final_norm = False,
            **transformer_kwargs
        ).cuda()

        # ppo related

        self.gae_use_accelerated = gae_use_accelerated
        self.gae_discount_factor = gae_discount_factor
        self.gae_lambda = gae_lambda

        self.ppo_eps_clip = ppo_eps_clip
        self.value_clip = value_clip
        self.policy_entropy_weight = policy_entropy_weight

        # pmpo related

        self.pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight
        self.pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight
        self.pmpo_reverse_kl = pmpo_reverse_kl

        # rewards related
        self.keep_reward_ema_stats = keep_reward_ema_stats
        self.reward_ema_decay = reward_ema_decay

        self.register_buffer('reward_quantile_filter', tensor(reward_quantile_filter), persistent = False)

        self.register_buffer('ema_returns_mean', tensor(0.))
        self.register_buffer('ema_returns_var', tensor(1.))

        # loss related

        self.flow_loss_normalizer = ut.LossNormalizer(1)
        self.reward_loss_normalizer = ut.LossNormalizer(multi_token_pred_len)
        has_discrete = (
            sum(num_discrete_actions) if isinstance(num_discrete_actions, (tuple, list))
            else num_discrete_actions
        ) > 0

        self.discrete_actions_loss_normalizer = (
            ut.LossNormalizer(multi_token_pred_len) if has_discrete else None
        )
        self.latent_flow_loss_weight = latent_flow_loss_weight

        self.register_buffer('reward_loss_weight', tensor(reward_loss_weight))
        self.register_buffer('discrete_action_loss_weight', tensor(discrete_action_loss_weight))
        self.register_buffer('continuous_action_loss_weight', tensor(continuous_action_loss_weight))

        assert self.reward_loss_weight.numel() in {1, multi_token_pred_len}
        assert self.discrete_action_loss_weight.numel() in {1, multi_token_pred_len}
        assert self.continuous_action_loss_weight.numel() in {1, multi_token_pred_len}

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    # types of parameters
    def reward_pred_parameters(self):
        """Get parameters from the reward prediction ensemble."""
        return self.to_reward_pred.parameters()

    def muon_parameters(self):
        return self.transformer.muon_parameters()

    def policy_head_online_parameters(self):
        return self.policy_head_online.parameters()

    # def policy_head_online_parameters(self):
    #     return [
    #         *self.policy_head_online.parameters(),
    #         *self.action_embedder.unembed_parameters() # includes the unembed from the action-embedder
    #     ]
    
    def policy_head_lagging_parameters(self):
        return [
            *self.policy_head_lagging.parameters(),
            *self.action_embedder.unembed_parameters() # includes the unembed from the action-embedder
        ]



    def value_head_parameters(self):
        return self.value_head.parameters()

    def parameter(self):
        params = super().parameters()

        if not exists(self.video_tokenizer):
            return params

        return list(set(params) - set(self.video_tokenizer.parameters()))

    # helpers for shortcut flow matching

    # evolutionary policy optimization - https://web3.arxiv.org/abs/2503.19037

    # interacting with env for experience

    @torch.no_grad()
    def Align_policy_Heads(self):
        for param_online, param_lagging in zip(self.policy_head_online.parameters(), self.policy_head_lagging.parameters()):
            param_lagging.data.copy_(param_online.data)



    def embedActions(
        self,
        signal_levels,
        align_dims_left_to = None
    ):

        # maybe create the action tokens

        if ut.exists(discrete_actions) or ut.exists(continuous_actions):
            assert self.action_embedder.has_actions
            assert self.num_agents == 1, 'only one agent allowed for now'

            action_tokens = self.action_embedder(
                discrete_actions = discrete_actions,
                discrete_action_types = discrete_action_types,
                continuous_actions = continuous_actions,
                continuous_action_types = continuous_action_types
            )

            # handle first timestep not having an associated past action

            if action_tokens.shape[1] == (time - 1):
                action_tokens = pad_at_dim(action_tokens, (1, 0), value = 0. , dim = 1)

            action_tokens = add('1 d, b t d', self.action_learned_embed, action_tokens)

        elif self.action_embedder.has_actions:
            action_tokens = torch.zeros_like(agent_tokens[:, :, 0:1])

        else:
            action_tokens = agent_tokens[:, :, 0:0] # else empty off agent tokens

        # main function, needs to be defined as such for shortcut training - additional calls for consistency loss


    def embedFlow(
        self,
        signal_levels,
        align_dims_left_to = None
    ):

        # signal and step size related input conforming

        if exists(signal_levels):
            if isinstance(signal_levels, int):
                signal_levels = tensor(signal_levels, device = self.device)

            if signal_levels.ndim == 0:
                signal_levels = repeat(signal_levels, '-> b', b = batch)

            if signal_levels.ndim == 1:
                signal_levels = repeat(signal_levels, 'b -> b t', t = time)

        if exists(step_sizes):
            if isinstance(step_sizes, int):
                step_sizes = tensor(step_sizes, device = self.device)

            if step_sizes.ndim == 0:
                step_sizes = repeat(step_sizes, '-> b', b = batch)

        if exists(step_sizes_log2):
            if isinstance(step_sizes_log2, int):
                step_sizes_log2 = tensor(step_sizes_log2, device = self.device)

            if step_sizes_log2.ndim == 0:
                step_sizes_log2 = repeat(step_sizes_log2, '-> b', b = batch)

        # handle step sizes -> step size log2

        assert not (exists(step_sizes) and exists(step_sizes_log2))

        if exists(step_sizes):
            step_sizes_log2_maybe_float = torch.log2(step_sizes)
            step_sizes_log2 = step_sizes_log2_maybe_float.long()

            assert (step_sizes_log2 == step_sizes_log2_maybe_float).all(), f'`step_sizes` must be powers of 2'

        # flow related

        assert not (exists(signal_levels) ^ exists(step_sizes_log2))

        is_inference = exists(signal_levels)
        no_shortcut_train = not is_inference

        return_pred_only = return_pred_only or latent_is_noised

        # if neither signal levels or step sizes passed in, assume training
        # generate them randomly for training

        if not is_inference:

            no_shortcut_train = sample_prob(self.prob_no_shortcut_train)

            if no_shortcut_train:
                # if no shortcut training, step sizes are just 1 and noising is all steps, where each step is 1 / d_min
                # in original shortcut paper, they actually set d = 0 for some reason, look into that later, as there is no mention in the dreamer paper of doing this

                step_sizes_log2 = zeros((batch,), device = device).long() # zero because zero is equivalent to step size of 1
                signal_levels = randint(0, self.max_steps, (batch, time), device = device)
            else:

                # now we follow eq (4)

                step_sizes_log2 = randint(1, self.num_step_sizes_log2, (batch,), device = device)
                num_step_sizes = 2 ** step_sizes_log2

                signal_levels = randint(0, self.max_steps, (batch, time), device=device) // num_step_sizes[:, None] * num_step_sizes[:, None] # times are discretized to step sizes


        # times is from 0 to 1

        times = self.get_times_from_signal_level(signal_levels)

        if not latent_is_noised:
            # get the noise

            noise = randn_like(latents)
            aligned_times = align_dims_left(times, latents)

            # noise from 0 as noise to 1 as data

            noised_latents = noise.lerp(latents, aligned_times)

        else:
            noised_latents = latents
    def embedTask(
        self,
        tasks,
        align_dims_left_to = None
    ):

        agent_tokens = repeat(self.agent_learned_embed, '... d -> b ... d', b = batch)

        if ut.exists(tasks):
            assert self.num_tasks > 0

            task_embeds = self.task_embed(tasks)
            agent_tokens = add('b ... d, b d', agent_tokens, task_embeds)

        # maybe evolution

        if ut.exists(latent_gene_ids):
            assert ut.exists(self.latent_genes)
            latent_genes = self.latent_genes[latent_gene_ids]

            agent_tokens = add('b ... d,  b d', agent_tokens, latent_genes)

        # handle agent tokens w/ actions and task embeds

        agent_tokens = repeat(agent_tokens, 'b ... d -> b t ... d', t = time)


    def rewardLoss(
        self,
        signal_levels):


        # reward loss---------------------------------------
        reward_loss = self.zero

        if exists(rewards):



            reward_targets, reward_loss_mask = create_multi_token_prediction_targets(two_hot_encoding[:, :-1], self.multi_token_pred_len)

            reward_targets = rearrange(reward_targets, 'b t mtp l -> b l t mtp')

            reward_losses = F.cross_entropy(reward_pred, reward_targets, reduction = 'none')

            reward_losses = reward_losses.masked_fill(~reward_loss_mask, 0.)

            if is_var_len:
                reward_loss = reward_losses[loss_mask_without_last].mean(dim = 0)
            else:
                reward_loss = reduce(reward_losses, '... mtp -> mtp', 'mean') # they sum across the prediction steps (mtp dimension) - eq(9)

    def actionLoss(
        self):

        # maybe autoregressive action loss

        discrete_action_loss = self.zero
        continuous_action_loss = self.zero

        if (
            self.num_agents == 1 and
            add_autoregressive_action_loss and
            time > 1,
            (exists(discrete_actions) or exists(continuous_actions))
        ):
            assert self.action_embedder.has_actions

            

            if exists(discrete_log_probs):
                discrete_log_probs = discrete_log_probs.masked_fill(~discrete_mask[..., None], 0.)

                if is_var_len:
                    discrete_action_losses = rearrange(-discrete_log_probs, 'mtp b t na -> b t na mtp')
                    discrete_action_loss = reduce(discrete_action_losses[loss_mask_without_last], '... mtp -> mtp', 'mean')
                else:
                    discrete_action_loss = reduce(-discrete_log_probs, 'mtp b t na -> mtp', 'mean')

            if exists(continuous_log_probs):
                continuous_log_probs = continuous_log_probs.masked_fill(~continuous_mask[..., None], 0.)

                if is_var_len:
                    continuous_action_losses = rearrange(-continuous_log_probs, 'mtp b t na -> b t na mtp')
                    continuous_action_loss = reduce(continuous_action_losses[loss_mask_without_last], '... mtp -> mtp', 'mean')
                else:
                    continuous_action_loss = reduce(-continuous_log_probs, 'mtp b t na -> mtp', 'mean')

        # reward loss  end---------------------------------------        

        #-------------------------------------------------flow loss end --------------------------------------------

    def forward(
        self,
        latentTokens,
        agent_tokens,
        register_tokens,
        flow_tokens,
        action_tokens = None,
        time_kv_cache = None,
    ):
        
        space_tokens = self.latents_to_spatial_tokens(latentTokens)
        #interleave tokens
        space_tokens, inverse_pack_space_per_latent = ut.pack_one(space_tokens, 'b t * d')
        if action_tokens is not None:
            action_tokens=action_tokens.cuda()

        if agent_tokens is not None:
            agent_tokens=agent_tokens.cuda()
        if flow_tokens.dim() < 4:
            flow_tokens = flow_tokens.unsqueeze(2)
        tokens, packed_tokens_shape = pack([flow_tokens, space_tokens, register_tokens, action_tokens, agent_tokens], 'b t * d')
        print("tokens tshape ",tokens.shape)
        # tokens, next_time_kv_cache = self.transformer.to(tokens)
        tokens, next_time_kv_cache = self.transformer(tokens, kv_cache = time_kv_cache, return_kv_cache = True)
        # unpack

        flow_tokens, space_tokens, register_tokens, action_tokens, agent_tokens = unpack(tokens, packed_tokens_shape, 'b t * d')

        # pooling
        
        space_tokens = inverse_pack_space_per_latent(space_tokens)
        
        # space_tokens = space_tokens.squeeze()
        print("st tshape ",space_tokens.shape)
        if space_tokens.dim() > 5:
            space_tokens = space_tokens.squeeze(4)#cheap hack for now
        pred = self.to_latent_pred(space_tokens)


        return pred, (agent_tokens, next_time_kv_cache)

