from __future__ import annotations
from math import ceil, log2
from dataclasses import dataclass
from typing import Optional, Tuple
import torchvision
import torch
from torch import is_tensor
from torch.nn import Module
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.models import VGG16_Weights
from collections import namedtuple
from adam_atan2_pytorch import MuonAdamAtan2
from functools import partial, wraps
from torch.nn import Linear
from torch.nn import Module, ModuleList, Embedding, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, full, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange
from einops import einsum, rearrange, repeat, reduce, pack, unpack
import einx
from einx import add, multiply
import torch.nn.functional as F
import random as random_module
from random import random
# try:
#     from torch.nn.attention.flex_attention import flex_attention, create_block_mask
#     if torch.cuda.is_available():
#         flex_attention = torch.compile(flex_attention)
# except ImportError:
#     print("that didnt work")
#     pass

def exists(v):
    return v is not None

# def create_multi_token_preddiction_targets(
#     t, # (b t ...)
#     steps_future,

# ): # (b t-1 steps ...), (b t-1 steps) - targets and the mask, where mask is False for padding

#     batch, seq_len, device = *t.shape[:2], t.device

#     batch_arange = arange(batch, device = device)
#     seq_arange = arange(seq_len, device = device)
#     steps_arange = arange(steps_future, device = device)

#     indices = add('t, steps -> t steps', seq_arange, steps_arange)
#     mask = indices < seq_len

#     batch_arange = rearrange(batch_arange, 'b -> b 1 1')

#     indices[~mask] = 0
#     mask = repeat(mask, 't steps -> b t steps', b = batch)

#     out = t[batch_arange, indices]

#     return out, mask
from torch.utils.data import Sampler

class RepeatEpisodeSampler(Sampler):
    def __init__(self, dataset, batch_size, samples_per_episode):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_episode = samples_per_episode
        self.n_episodes = len(dataset)
        
    def __iter__(self):
        # Repeat each episode index multiple times
        indices = []
        for ep_idx in range(self.n_episodes):
            indices.extend([ep_idx] * self.samples_per_episode)
        
        # Shuffle so different episodes mix in batches
        random_module.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.n_episodes * self.samples_per_episode
    

    
def save_random_recon_comparison(video, recon_video, step, save_dir="recon_frames"): 
    """
    video, recon_video: (b, c, t, h, w)
    Saves 1 random frame from the first sample in the batch,
    as [original | reconstruction] in a single image.
    """

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # take first sample in batch
        vid_orig  = video[0]        # (c, t, h, w)
        vid_recon = recon_video[0]  # (c, t, h, w)

        _, T, _, _ = vid_orig.shape
        t_idx = random_module.randint(0, T - 1)

        orig_frame  = vid_orig[:, t_idx]   # (c, h, w)
        recon_frame = vid_recon[:, t_idx]  # (c, h, w)

        # detach + move to cpu
        orig_frame  = orig_frame.detach().cpu()
        recon_frame = recon_frame.detach().cpu()

        # assume data may be in [-1, 1] → map to [0, 1]
        def to_01(x):
            if x.min() < 0:
                x = (x.clamp(-1, 1) + 1) / 2
            else:
                x = x.clamp(0, 1)
            return x

        orig_frame  = to_01(orig_frame)
        recon_frame = to_01(recon_frame)

        # make [orig, recon] grid: 2 x (c,h,w)
        grid = torch.stack([orig_frame, recon_frame], dim=0)

        path = os.path.join(save_dir, f"step_{step:06d}_t{t_idx:02d}.png")
        torchvision.utils.save_image(grid, path, nrow=2)
        # left = original, right = reconstruction


MaybeTensor = Tensor | None
# import logg ing
# logg ing.basicConfig(
#     level=log ging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='app.log',
#     filemode='w'  # 'w' to overwrite, 'a' to append
# )
from torch.nested import nested_tensor
@dataclass
class Experience:
    latents: Tensor
    video: MaybeTensor = None
    proprio: MaybeTensor = None
    agent_embed: MaybeTensor = None
    rewards: Tensor | None = None
    actions: tuple[MaybeTensor, MaybeTensor] | None = None
    log_probs: tuple[MaybeTensor, MaybeTensor] | None = None
    old_action_unembeds: tuple[MaybeTensor, MaybeTensor] | None = None
    values: MaybeTensor = None
    step_size: int | None = None
    lens: MaybeTensor = None
    is_truncated: MaybeTensor = None
    agent_index: int = 0
    is_from_world_model: bool = True
    logits: MaybeTensor = None

    def cpu(self):
        """Move all tensor fields to CPU"""
        cpu_fields = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, torch.Tensor):
                cpu_fields[field_name] = field_value.cpu()
            elif isinstance(field_value, tuple):
                cpu_fields[field_name] = tuple(
                    t.cpu() if isinstance(t, torch.Tensor) else t 
                    for t in field_value
                )
            else:
                cpu_fields[field_name] = field_value
        return Experience(**cpu_fields)
    def cuda(self, device=None):
        """Move all tensor fields to CUDA"""
        cuda_fields = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, torch.Tensor):
                cuda_fields[field_name] = field_value.cuda(device)
            elif isinstance(field_value, tuple):
                cuda_fields[field_name] = tuple(
                    t.cuda(device) if isinstance(t, torch.Tensor) else t 
                    for t in field_value
                )
            else:
                cuda_fields[field_name] = field_value
        return Experience(**cuda_fields)
    
    def to(self, device):
        """Move all tensor fields to specified device"""
        device_fields = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, torch.Tensor):
                device_fields[field_name] = field_value.to(device)
            elif isinstance(field_value, tuple):
                device_fields[field_name] = tuple(
                    t.to(device) if isinstance(t, torch.Tensor) else t 
                    for t in field_value
                )
            else:
                device_fields[field_name] = field_value
        return Experience(**device_fields)
def combine_experiences(
    exps: list[Experiences]
) -> Experience:

    assert len(exps) > 0

    # set lens if not there

    for exp in exps:
        latents = exp.latents
        batch, time, device = *latents.shape[:2], latents.device

        if not exists(exp.lens):
            exp.lens = full((batch,), time, device = device)

        if not exists(exp.is_truncated):
            exp.is_truncated = full((batch,), True, device = device)

    # convert to dictionary

    exps_dict = [asdict(exp) for exp in exps]

    values, tree_specs = zip(*[tree_flatten(exp_dict) for exp_dict in exps_dict])

    tree_spec = first(tree_specs)

    all_field_values = list(zip(*values))

    # an assert to make sure all fields are either all tensors, or a single matching value (for step size, agent index etc) - can change this later

    assert all([
        all([is_tensor(v) for v in field_values]) or len(set(field_values)) == 1
        for field_values in all_field_values
    ])

    concatted = []

    for field_values in all_field_values:

        if is_tensor(first(field_values)):

            field_values = pad_tensors_at_dim_to_max_len(field_values, dims = (1, 2))

            new_field_value = cat(field_values)
        else:
            new_field_value = first(list(set(field_values)))

        concatted.append(new_field_value)

    # return experience

    concat_exp_dict = tree_unflatten(concatted, tree_spec)

    return Experience(**concat_exp_dict)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def has_at_least_one(*bools):
    return sum([*map(int, bools)]) > 0

def ensure_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

def divisible_by(num, den):
    return (num % den) == 0

def sample_prob(prob):
    return random() < prob

def is_power_two(num):
    return log2(num).is_integer()

# tensor helpers

def is_empty(t):
    return t.numel() == 0

def lens_to_mask(t, max_len = None):
    if not exists(max_len):
        max_len = t.amax().item()

    device = t.device
    seq = torch.arange(max_len, device = device)

    return einx.less('j, i -> i j', seq, t)
def ramp_weight(times, slope = 0.9, intercept = 0.1):
    # equation (8) paper, their "ramp" loss weighting
    return slope * times + intercept
def masked_mean(t, mask = None):
    if not exists(mask):
        return t.mean()

    if not mask.any():
        return t[mask].sum()

    return t[mask].mean()

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def mean_log_var_to_distr(
    mean_log_var: Tensor
) -> Normal:

    mean, log_var = mean_log_var.unbind(dim = -1)
    std = (0.5 * log_var).exp()
    return Normal(mean, std)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return tensors[0]

    return cat(tensors, dim = dim)

def safe_squeeze_first(t):
    if not exists(t):
        return None

    if t.shape[0] != 1:
        return t

    return rearrange(t, '1 ... -> ...')

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(
    t,
    temperature = 1.,
    dim = -1,
    keepdim = False,
    eps = 1e-10
):
    noised = (t / max(temperature, eps)) + gumbel_noise(t)
    return noised.argmax(dim = dim, keepdim = keepdim)

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return first(unpack(out, packed_shape, inv_pattern))

    return packed, inverse

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_to_len(t, target_len, *, dim):
    curr_len = t.shape[dim]

    if curr_len >= target_len:
        return t

    return pad_at_dim(t, (0, target_len - curr_len), dim = dim)

def pad_tensors_at_dim_to_max_len(
    tensors: list[Tensor],
    dims: tuple[int, ...]
):
    for dim in dims:
        if dim >= first(tensors).ndim:
            continue

        max_time = max([t.shape[dim] for t in tensors])
        tensors = [pad_to_len(t, max_time, dim = dim) for t in tensors]

    return tensors

def align_dims_left(t, aligned_to):
    shape = t.shape
    num_right_dims = aligned_to.ndim - t.ndim

    if num_right_dims < 0:
        return

    return t.reshape(*shape, *((1,) * num_right_dims))
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

def create_multi_token_prediction_targets(
    t, # (b t ...)
    steps_future,

): # (b t-1 steps ...), (b t-1 steps) - targets and the mask, where mask is False for padding

    batch, seq_len, device = *t.shape[:2], t.device

    batch_arange = arange(batch, device = device)
    seq_arange = arange(seq_len, device = device)
    steps_arange = arange(steps_future, device = device)

    indices = add('t, steps -> t steps', seq_arange, steps_arange)
    mask = indices < seq_len

    batch_arange = rearrange(batch_arange, 'b -> b 1 1')

    indices[~mask] = 0
    mask = repeat(mask, 't steps -> b t steps', b = batch)

    out = t[batch_arange, indices]

    return out, mask

# loss related

class LossNormalizer(Module):

    # the authors mentioned the need for loss normalization in the dynamics transformer

    def __init__(
        self,
        num_losses: int,
        beta = 0.95,
        eps = 1e-6
    ):
        super().__init__()
        self.register_buffer('exp_avg_sq', torch.ones(num_losses))
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        losses: Tensor | list[Tensor] | dict[str, Tensor],
        update_ema = None
    ):
        exp_avg_sq, beta = self.exp_avg_sq, self.beta
        update_ema = default(update_ema, self.training)

        # get the rms value - as mentioned at the end of section 3 in the paper

        rms = exp_avg_sq.sqrt()

        if update_ema:
            decay = 1. - beta

            # update the ema

            exp_avg_sq.lerp_(losses.detach().square(), decay)

        # then normalize

        assert losses.numel() == rms.numel()

        normed_losses = losses / rms.clamp(min = self.eps)

        return normed_losses

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        sampled_frames = 1
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]
        self.sampled_frames = sampled_frames

    def forward(
        self,
        pred,
        data,
    ):
        batch, device, is_video = pred.shape[0], pred.device, pred.ndim == 5

        vgg, = self.vgg
        vgg = vgg.to(data.device)

        # take care of sampling random frames of the video

        if is_video:
            pred, data = tuple(rearrange(t, 'b c t ... -> b t c ...') for t in (pred, data))

            # batch randperm

            batch_randperm = randn(pred.shape[:2], device = pred.device).argsort(dim = -1)
            rand_frames = batch_randperm[..., :self.sampled_frames]

            batch_arange = arange(batch, device = device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')

            pred, data = tuple(t[batch_arange, rand_frames] for t in (pred, data))

            # fold sampled frames into batch

            pred, data = tuple(rearrange(t, 'b t c ... -> (b t) c ...') for t in (pred, data))

        pred_embed, embed = tuple(vgg(t) for t in (pred, data))

        return F.mse_loss(embed, pred_embed)

def ramp_weight(times, slope = 0.9, intercept = 0.1):
    # equation (8) paper, their "ramp" loss weighting
    return slope * times + intercept

# reinforcement learning related

# rewards

class SymExpTwoHot(Module):
    def __init__(
        self,
        reward_range = (-20., 20.),
        num_bins = 255,
        learned_embedding = False,
        dim_embed = None,
    ):
        super().__init__()

        min_value, max_value = reward_range
        values = linspace(min_value, max_value, num_bins)
        values = values.sign() * (torch.exp(values.abs()) - 1.)

        self.reward_range = reward_range
        self.num_bins = num_bins
        self.register_buffer('bin_values', values)
        self.bin_values=self.bin_values.cuda()

        # take care of a reward embedding
        # for an improvisation where agent tokens can also see the past rewards - it makes sense that this information should not be thrown out, a la Decision Transformer

        self.learned_embedding = learned_embedding

        if learned_embedding:
            assert exists(dim_embed)
            self.bin_embeds = nn.Embedding(num_bins, dim_embed)

    @property
    def device(self):
        return self.bin_values.device

    def embed(
        self,
        two_hot_encoding,
    ):
        assert self.learned_embedding, f'can only embed if `learned_embedding` is True'

        weights, bin_indices = two_hot_encoding.topk(k = 2, dim = -1)

        two_embeds = self.bin_embeds(bin_indices)

        return einsum(two_embeds, weights, '... two d, ... two -> ... d')

    def bins_to_scalar_value(
        self,
        logits, # (... l)
        normalize = False
    ):
        two_hot_encoding = logits.softmax(dim = -1) if normalize else logits
        return einsum(two_hot_encoding, self.bin_values, '... l, l -> ...')

    def forward(
        self,
        values
    ):
        bin_values = self.bin_values
        min_bin_value, max_bin_value = self.bin_values[0], self.bin_values[-1]

        values, inverse_pack = pack_one(values, '*')
        values=values.cuda()
        num_values = values.shape[0]

        values = values.clamp(min = min_bin_value, max = max_bin_value)

        indices = torch.searchsorted(self.bin_values, values)

        # fetch the closest two indices (two-hot encoding)

        left_indices = (indices - 1).clamp(min = 0)
        right_indices = left_indices + 1

        left_indices, right_indices = tuple(rearrange(t, '... -> ... 1') for t in (left_indices, right_indices))

        # fetch the left and right values for the consecutive indices

        left_values = self.bin_values[left_indices]
        right_values = self.bin_values[right_indices]

        # calculate the left and right values by the distance to the left and right

        values = rearrange(values, '... -> ... 1')
        total_distance = right_values - left_values

        left_logit_value = (right_values - values) / total_distance
        right_logit_value = 1. - left_logit_value

        # set the left and right values (two-hot)

        encoded = torch.zeros((num_values, self.num_bins), device = self.device)

        encoded.scatter_(-1, left_indices, left_logit_value)
        encoded.scatter_(-1, right_indices, right_logit_value)

        return inverse_pack(encoded, '* l')

# action related

ActionEmbeds = namedtuple('ActionEmbed', ('discrete', 'continuous'))

class ActionEmbedder(Module):
    def __init__(
        self,
        dim,
        *,
        num_discrete_actions: int | tuple[int, ...] = 0,

        can_unembed = False,
        unembed_dim = None,
        num_unembed_preds = 1,
        squeeze_unembed_preds = True # will auto-squeeze if prediction is just 1
    ):
        super().__init__()

        # handle discrete actions

        num_discrete_actions = tensor(ensure_tuple(num_discrete_actions))
        print("num_discrete_actions", num_discrete_actions, num_discrete_actions.shape)

        total_discrete_actions = num_discrete_actions.sum().item()

        self.num_discrete_action_types = len(num_discrete_actions)
        self.discrete_action_embed = Embedding(total_discrete_actions, dim)

        self.register_buffer('num_discrete_actions', num_discrete_actions, persistent = False)



        # defaults

        self.register_buffer('default_discrete_action_types', arange(self.num_discrete_action_types), persistent = False)


        # calculate offsets

        offsets = F.pad(num_discrete_actions.cumsum(dim = -1), (1, -1), value = 0)
        self.register_buffer('discrete_action_offsets', offsets, persistent = False)

        # unembedding

        self.can_unembed = can_unembed

        self.num_unembed_preds = num_unembed_preds
        self.squeeze_unembed_preds = squeeze_unembed_preds

        if not can_unembed:
            return

        unembed_dim = default(unembed_dim, dim)
        self.discrete_action_unembed = Parameter(torch.randn(total_discrete_actions, num_unembed_preds, unembed_dim) * 1e-2)

        discrete_action_index = arange(total_discrete_actions)

        padded_num_discrete_actions = F.pad(num_discrete_actions, (1, 0), value = 0)
        exclusive_cumsum = padded_num_discrete_actions.cumsum(dim = -1)

        discrete_action_mask = (
            einx.greater_equal('j, i -> i j', discrete_action_index, exclusive_cumsum[:-1]) &
            einx.less('j, i -> i j', discrete_action_index, exclusive_cumsum[1:])
        )

        self.register_buffer('discrete_action_mask', discrete_action_mask, persistent = False)


    def embed_parameters(self):
        return next(self.discrete_action_embed.parameters())


    def unembed_parameters(self):
        return self.discrete_action_unembed

    @property
    def device(self):
        return self.discrete_action_offsets.device

    @property
    def has_actions(self):
        return self.num_discrete_action_types > 0 or self.num_continuous_action_types > 0

    def cast_action_types(
        self,
        action_types = None
    ):
        if exists(action_types) and not is_tensor(action_types):
            if isinstance(action_types, int):
                action_types = (action_types,)

            action_types = tensor(action_types, device = self.device)

        return action_types

    def unembed(
        self,
        embeds,                          # (... d)
        discrete_action_types = None,    # (na)
        continuous_action_types = None,  # (na)
        return_split_discrete = False,
        pred_head_index: int | Tensor | None = None

    ):  # (... discrete_na), (... continuous_na 2)

        device = embeds.device

        assert self.can_unembed, 'can only unembed for predicted discrete and continuous actions if `can_unembed = True` is set on init'

        # handle only one prediction head during inference

        if exists(pred_head_index) and isinstance(pred_head_index, int):
            pred_head_index = tensor(pred_head_index, device = device)

        # if pred_head_index given as a solo int, just assume we want to squeeze out the prediction head dimension

        squeeze_one_pred_head = exists(pred_head_index) and pred_head_index.ndim == 0

        # get action types
        discrete_action_types = self.cast_action_types(discrete_action_types)
        # discrete actions

        discrete_action_logits = None

        if self.num_discrete_action_types > 0:
            discrete_action_unembed = self.discrete_action_unembed

            if exists(discrete_action_types):
                discrete_action_mask = self.discrete_action_mask[discrete_action_types].any(dim = 0)

                discrete_action_unembed = discrete_action_unembed[discrete_action_mask]

            if exists(pred_head_index):
                discrete_action_unembed = discrete_action_unembed.index_select(1, pred_head_index)

            discrete_action_logits = einsum(embeds, discrete_action_unembed, '... d, na mtp d -> mtp ... na')

            if self.squeeze_unembed_preds or squeeze_one_pred_head:
                discrete_action_logits = safe_squeeze_first(discrete_action_logits)

        # whether to split the discrete action logits by the number of actions per action type

        if exists(discrete_action_logits) and return_split_discrete:

            split_sizes = self.num_discrete_actions[discrete_action_types] if exists(discrete_action_types) else self.num_discrete_actions

            discrete_action_logits = discrete_action_logits.split(split_sizes.tolist(), dim = -1)


        return discrete_action_logits

    def sample(
        self,
        embed,
        discrete_temperature = 1.,

        pred_head_index: int | Tensor | None = None,
        squeeze = True,
        **kwargs
    ):

        discrete_logits = self.unembed(embed, return_split_discrete = True, pred_head_index = pred_head_index, **kwargs)

        sampled_discrete = sampled_continuous = None

        if exists(discrete_logits):
            sampled_discrete = []

            for one_discrete_logits in discrete_logits:
                sampled_discrete.append(gumbel_sample(one_discrete_logits, temperature = discrete_temperature, keepdim = True))

            sampled_discrete = cat(sampled_discrete, dim = -1)

        return sampled_discrete

    def log_probs(
        self,
        embeds,                          # (... d)
        discrete_targets = None,         # (... na)
        continuous_targets = None,       # (... na)
        discrete_action_types = None,    # (na)
        continuous_action_types = None,  # (na)
        pred_head_index: int | Tensor | None = None,
        parallel_discrete_calc = None,
        return_entropies = False
    ):
        parallel_discrete_calc = default(parallel_discrete_calc, exists(discrete_targets) and discrete_targets.shape[-1] > 1)

        discrete_action_logits = self.unembed(embeds, pred_head_index = pred_head_index, discrete_action_types = discrete_action_types, continuous_action_types = continuous_action_types, return_split_discrete = True)

        # discrete

        discrete_log_probs = None
        discrete_entropies = None

        if exists(discrete_targets):

            if parallel_discrete_calc:
                # use nested tensors

                jagged_dims = tuple(t.shape[-1] for t in discrete_action_logits)

                discrete_action_logits = cat(discrete_action_logits, dim = -1)

                discrete_action_logits, inverse_pack_lead_dims = pack_one(discrete_action_logits, '* l')
                batch = discrete_action_logits.shape[0]

                discrete_action_logits = rearrange(discrete_action_logits, 'b l -> (b l)')

                # to nested tensor

                nested_logits = nested_tensor(discrete_action_logits.split(jagged_dims * batch), layout = torch.jagged, device = self.device, requires_grad = True)

                prob = nested_logits.softmax(dim = -1)

                log_probs = log(prob)

                # maybe entropy

                if return_entropies:
                    discrete_entropies = (-prob * log_probs).sum(dim = -1, keepdim = True)
                    discrete_entropies = cat(discrete_entropies.unbind())
                    discrete_entropies = rearrange(discrete_entropies, '(b na) -> b na', b = batch)

                    discrete_entropies = inverse_pack_lead_dims(discrete_entropies, '* na')

                # back to regular tensor

                log_probs = cat(log_probs.unbind())
                log_probs = rearrange(log_probs, '(b l) -> b l', b = batch)

                log_probs = inverse_pack_lead_dims(log_probs)

                # get indices to gather

                discrete_action_types = default(discrete_action_types, self.default_discrete_action_types)

                num_discrete_actions = self.num_discrete_actions[discrete_action_types]

                offset = F.pad(num_discrete_actions.cumsum(dim = -1), (1, -1), value = 0)
                log_prob_indices = discrete_targets + offset

                # gather

                discrete_log_probs = log_probs.gather(-1, log_prob_indices)

            else:
                discrete_log_probs = []
                discrete_entropies = []

                for one_discrete_action_logit, one_discrete_target in zip(discrete_action_logits, discrete_targets.unbind(dim = -1)):

                    one_discrete_probs = one_discrete_action_logit.softmax(dim = -1)
                    one_discrete_log_probs = log(one_discrete_probs)
                    one_discrete_target = rearrange(one_discrete_target, '... -> ... 1')

                    log_prob = one_discrete_log_probs.gather(-1, one_discrete_target)
                    discrete_log_probs.append(log_prob)

                    if return_entropies:
                        entropy = (-one_discrete_probs * one_discrete_log_probs).sum(dim = -1)
                        discrete_entropies.append(entropy)

                discrete_log_probs = cat(discrete_log_probs, dim = -1)

                if return_entropies:
                    discrete_entropies = stack(discrete_entropies, dim = -1)

        # continuous

        log_probs = discrete_log_probs

        if not return_entropies:
            return log_probs

        entropies = (discrete_entropies)

        return log_probs, entropies

    def kl_div(
        self,
        src: tuple[MaybeTensor, MaybeTensor],
        tgt: tuple[MaybeTensor, MaybeTensor],
        reduce_across_num_actions = True
    ) -> tuple[MaybeTensor, MaybeTensor]:

        src_discrete, src_continuous = src
        tgt_discrete, tgt_continuous = tgt

        discrete_kl_div = None

        # split discrete if it is not already (multiple discrete actions)

        if exists(src_discrete):

            discrete_split = self.num_discrete_actions.tolist()

            if is_tensor(src_discrete):
                src_discrete = src_discrete.split(discrete_split, dim = -1)

            if is_tensor(tgt_discrete):
                tgt_discrete = tgt_discrete.split(discrete_split, dim = -1)

            discrete_kl_divs = []

            for src_logit, tgt_logit in zip(src_discrete, tgt_discrete):

                src_log_probs = src_logit.log_softmax(dim = -1)
                tgt_prob = tgt_logit.softmax(dim = -1)

                one_discrete_kl_div = F.kl_div(src_log_probs, tgt_prob, reduction = 'none')

                discrete_kl_divs.append(one_discrete_kl_div.sum(dim = -1))

            discrete_kl_div = stack(discrete_kl_divs, dim = -1)

        # calculate kl divergence for continuous



        # maybe reduce

        if reduce_across_num_actions:
            if exists(discrete_kl_div):
                discrete_kl_div = discrete_kl_div.sum(dim = -1)


                continuous_kl_div = continuous_kl_div.sum(dim = -1)

        return discrete_kl_div

    def forward(
        self,
        *,
        discrete_actions = None,         # (... na)

        discrete_action_types = None    # (na)

    ):

        discrete_embeds = None

        if exists(discrete_actions):

            discrete_action_types = default(discrete_action_types, self.default_discrete_action_types)

            discrete_action_types = self.cast_action_types(discrete_action_types)

            offsets = self.discrete_action_offsets[discrete_action_types]

            assert offsets.shape[-1] == discrete_actions.shape[-1], 'mismatched number of discrete actions'

            # offset the discrete actions based on the action types passed in (by default all discrete actions) and the calculated offset

            discrete_actions_offsetted = add('... na, na', discrete_actions, offsets)
            discrete_embeds = self.discrete_action_embed(discrete_actions_offsetted)


        # return not pooled

        rety = ActionEmbeds(discrete_embeds, None)
        return discrete_embeds

        # handle sum pooling, which is what they did in the paper for all the actions


# generalized advantage estimate


# rotary embeddings for time

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

# multi-head rmsnorm

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

# naive attend

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

# flex attention related and factory function for attend depending on whether on cuda + flex attention available

def block_mask_causal(block_size):

    def inner(b, h, q, k):
        bq = q // block_size
        bk = k // block_size
        return bq >= bk

    return inner

def special_token_mask(q, k, seq_len, num_tokens, special_attend_only_itself = False):
    bq = q % seq_len
    bk = k % seq_len

    is_special_start_index = seq_len - num_tokens

    q_is_special = q >= is_special_start_index
    k_is_special = k >= is_special_start_index

    if special_attend_only_itself:
        out = ~(q_is_special & ~k_is_special) # modality attends to everything, but latent can only attend to itself (proposed attention pattern for encoder of video tokenizer)
    else:
        out = ~(~q_is_special & k_is_special) # modality cannot attend to agent tokens

    return out

def block_mask_special_tokens_right(
    seq_len,
    num_tokens,
    special_attend_only_itself = False
):
    def inner(b, h, q, k):
        return special_token_mask(q, k, seq_len, num_tokens, special_attend_only_itself)
    return inner

def compose_mask(mask1, mask2):
    def inner(b, h, q, k):
        return mask1(b, h, q, k) & mask2(b, h, q, k)

    return inner

def block_mask_noop(b, h, q, k):
    return b >= 0

def score_mod_softclamp(value):
    def inner(sim, b, h, q, k):
        if not exists(value):
           return sim

        sim = sim / value
        sim = torch.tanh(sim)
        sim = sim * value
        return sim

    return inner

# factory for attend function

def get_attend_fn(
    use_flex,
    seq_len,
    k_seq_len,
    causal = False,
    causal_block_size = 1,
    softclamp_value = 50.,
    num_special_tokens = 0,             # special tokens are latents / agents
    block_size_per_special = None,      # defaults to k_seq_len
    special_attend_only_itself = False, # by default, modality only attends to itself while special sees everything, but if turned True, will be the inverse - special can only attend to itself but modality can attend everything
    device = None
):
    block_size_per_special = default(block_size_per_special, k_seq_len)

    if use_flex:
        # flex pathway

        block_mask_fn = block_mask_causal(causal_block_size) if causal else block_mask_noop

        if num_special_tokens > 0:
            special_block_mask = block_mask_special_tokens_right(block_size_per_special, num_special_tokens, special_attend_only_itself) # NOTE: ASK LR
            block_mask_fn = compose_mask(block_mask_fn, special_block_mask)

        block_mask = create_block_mask(block_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = k_seq_len)

        score_mod = score_mod_softclamp(softclamp_value)
        attend_fn = partial(flex_attention, block_mask = block_mask, score_mod = score_mod, enable_gqa = True)
    else:
        # naive pathway

        mask = None
        if num_special_tokens > 0:
            q_seq = torch.arange(seq_len, device = device)[:, None]
            k_seq = torch.arange(k_seq_len, device = device)[None, :]

            mask = special_token_mask(q_seq, k_seq, block_size_per_special, num_special_tokens, special_attend_only_itself)

        attend_fn = partial(naive_attend, causal = causal, causal_block_size = causal_block_size, mask = mask, softclamp_value = softclamp_value)

    return attend_fn


import os

def pathExist(path):

    return os.path.exists(path)


LinearNoBias = partial(Linear, bias = False)