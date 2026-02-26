


import math
from math import ceil, log2
from random import random
import random as random_module
from contextlib import nullcontext
from collections import namedtuple
from functools import partial, wraps
from dataclasses import dataclass, asdict
from . import utils as ut
from . import transformer as trafo
import torch
import torch.nn.functional as F
from torch.nested import nested_tensor
from torch.distributions import Normal, kl
from torch.nn import Module, ModuleList, Embedding, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, full, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange
from torch.utils._pytree import tree_flatten, tree_unflatten
import einx
import torchvision
from torchvision.models import VGG16_Weights
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from torch.optim import Optimizer
from adam_atan2_pytorch import MuonAdamAtan2
from einops.layers.torch import Rearrange, Reduce
from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import create_mlp

from hyper_connections import get_init_and_expand_reduce_stream_functions

from assoc_scan import AssocScan

import math

from vit_pytorch.vit_with_decorr import DecorrelationLoss

from math import ceil, log2
from random import random
import random as random_module
from contextlib import nullcontext
from collections import namedtuple
from functools import partial, wraps
from dataclasses import dataclass, asdict
from . import utils as ut
from . import transformer as trafo
import torch
import torch.nn.functional as F
from torch.nested import nested_tensor
from torch.distributions import Normal, kl
from torch.nn import Module, ModuleList, Embedding, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, full, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange
from torch.utils._pytree import tree_flatten, tree_unflatten
import einx
import torchvision
from torchvision.models import VGG16_Weights
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from torch.optim import Optimizer
from adam_atan2_pytorch import MuonAdamAtan2
from einops.layers.torch import Rearrange, Reduce
from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import create_mlp

from hyper_connections import get_init_and_expand_reduce_stream_functions

from assoc_scan import AssocScan
def to_bnd(x: torch.Tensor) -> torch.Tensor:
    # (..., D) -> (1, N, D)
    d = x.shape[-1]
    return x.reshape(-1, d).unsqueeze(0)
def read_number(filepath="number.txt"):
    with open(filepath, "r") as f:
        content = f.read().strip()
    
    try:
        number = int(content)
    except ValueError:
        raise ValueError(f"File content '{content}' is not a valid integer.")
    
    if not 0 <= number <= 100:
        raise ValueError(f"Number {number} is out of range (0-100).")
    
    return number

def interpolate(input_value, start, stop):
    return start + (stop - start) * (input_value / 100)

class Encoder(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        patch_size,
        image_height = None,
        image_width = None,
        num_latent_tokens = 4,
        encoder_depth = 6,

        time_block_every = 4,
        attn_kwargs: dict = dict(),
        attn_dim_head = 32,
        attn_heads = 8,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        decoder_pos_mlp_depth = 2,
        channels = 3,
        per_image_patch_mask_prob = (0., 0.4), # probability of patch masking appears to be per image probabilities drawn uniformly between 0. and 0.9 - if you are a phd student and think i'm mistakened, please open an issue
        lpips_loss_network: Module | None = None,
        lpips_loss_weight = 0.2,
        nd_rotary_kwargs: dict = dict(
            rope_min_freq = 1.,
            rope_max_freq = 10000.,
            rope_p_zero_freqs = 0.
        ),
        num_residual_streams = 1
    ):
        super().__init__()

        self.patch_size = patch_size

        # special tokens

        assert num_latent_tokens >= 1
        self.num_latent_tokens = num_latent_tokens
        self.latent_tokens = Parameter(randn(num_latent_tokens, dim) * 1e-2)
        self.stepCount=0
        # hyper connections

        hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)
        self.mask_token = Parameter(torch.randn(dim) * 0.02)
        # mae masking - Kaiming He paper from long ago

        self.per_image_patch_mask_prob = per_image_patch_mask_prob
        self.mask_token = Parameter(randn(dim) * 1e-2)

        # patch and unpatch

        dim_patch = channels * patch_size ** 2


        self.encoder_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)



        self.patch_to_tokens = Sequential(
            Rearrange('b c t (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            Linear(dim_patch, dim)
        )


        self.patch_to_tokens_postcalc = Sequential(
            Linear(dim, dim),
            nn.SiLU(),
            Linear(dim, dim),
            nn.SiLU(),
            Linear(dim, dim)
            )






        # encoder space / time transformer

        self.encoder_transformer = trafo.AxialSpaceTimeTransformer(
            dim = dim,
            depth = encoder_depth,
            attn_dim_head = attn_dim_head,
            attn_softclamp_value = attn_softclamp_value,
            time_block_every = time_block_every,
            num_special_spatial_tokens = num_latent_tokens,
            num_residual_streams = num_residual_streams,
            final_norm = True
        )

        # latents

        self.encoded_to_latents = Sequential(
            ut.LinearNoBias(dim, dim_latent),
            nn.Tanh(),
        )

        self.latents_to_decoder = ut.LinearNoBias(dim_latent, dim)

        # decoder

        self.image_height = image_height
        self.image_width = image_width

        # parameterize the decoder positional embeddings for MAE style training so it can be resolution agnostic

        self.to_decoder_pos_emb = create_mlp(
            dim_in = 2,
            dim = dim * 2,
            dim_out = dim,
            depth = decoder_pos_mlp_depth,
        )

        # decoder transformer

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

        self.has_lpips_loss = lpips_loss_weight > 0.
        self.lpips_loss_weight = lpips_loss_weight

        if self.has_lpips_loss:
            self.lpips = ut.LPIPSLoss(lpips_loss_network)

        # In Encoder.__init__, add:
        if image_height is not None and image_width is not None:
            num_patches_h = image_height // patch_size
            num_patches_w = image_width // patch_size
            num_patches = num_patches_h * num_patches_w
            
            self.spatial_pos_embed = Parameter(
                torch.randn(1, 1, num_patches, dim) * 0.02
            )


    @property
    def device(self):
        return self.zero.device

    def muon_parameters(self):
        return [
            *self.encoder_transformer.muon_parameters(),
            *self.decoder_transformer.muon_parameters()
        ]


    def forward(
        self,
        video, # (b c t h w) 
        return_latents = False,
        mask_patches = None,
        return_all_losses = False,
        inference = False,

    ):
        # w = next(self.encoder_transformer.parameters())
        # print("early weight value:", w.view(-1)[0].item())
        # progress=read_number()
        # divFactor=interpolate(progress,1.0,255.0)
        # print(f"mean: {video.mean():.2f}, max: {video.max():.2f}, min: {video.min():.2f}")
        # exit()
        if inference:
            self.encoder_transformer.eval()
        else:
            self.encoder_transformer.train()
            
        self.stepCount+=1
        batch, _, time, height, width = video.shape
        patch_size, device = self.patch_size, video.device

        assert ut.divisible_by(height, patch_size) and ut.divisible_by(width, patch_size)

        # to tokens
        video = rearrange(video, 'b c t h w -> (b t) c h w')
        video = self.encoder_conv(video)
        video = rearrange(video, '(b t) c h w -> b c t h w', t=time)
        # video = video.float() / divFactor

        tokens = self.patch_to_tokens(video)

        tokens = self.patch_to_tokens_postcalc(tokens)+tokens

        # get some dimensions

        num_patch_height, num_patch_width, _ = tokens.shape[-3:]

        # Add spatial positional encoding
        tokens_with_pos = rearrange(tokens, 'b t h w d -> b t (h w) d')
        tokens_with_pos = tokens_with_pos + self.spatial_pos_embed  # ← ADD THIS
        tokens = rearrange(tokens_with_pos, 'b t (h w) d -> b t h w d', 
                        h=num_patch_height, w=num_patch_width)

        # masking

        mask_patches = ut.default(mask_patches, self.training)

        if mask_patches:
            min_mask_prob, max_mask_prob = self.per_image_patch_mask_prob

            mask_prob = torch.empty(tokens.shape[:2], device = tokens.device).uniform_(min_mask_prob, max_mask_prob) # (b t)

            mask_prob = repeat(mask_prob, 'b t -> b t vh vw', vh = tokens.shape[2], vw = tokens.shape[3])
            mask_patch = torch.bernoulli(mask_prob) == 1.

            tokens = einx.where('..., d, ... d', mask_patch, self.mask_token, tokens)

        # pack space

        tokens, inverse_pack_space = ut.pack_one(tokens, 'b t * d')

        # add the latent

        latents = repeat(self.latent_tokens, 'n d -> b t n d', b = tokens.shape[0], t = tokens.shape[1])
        # print(latents.shape)
        # print(tokens.shape)
        tokens, packed_latent_shape = pack((tokens, latents), 'b t * d')

        # encoder attention
        # print(tokens.shape)
        tokens, interm = self.encoder_transformer(tokens, return_intermediates = True)

        # latent bottleneck

        tokens, latents = unpack(tokens, packed_latent_shape, 'b t * d')

        latents = self.encoded_to_latents(latents)


        # recon_video = self.decode(latents, height = height, width = width)
        # if self.stepCount% 300==0 or self.stepCount==1:
        #     ut.save_random_recon_comparison(video, recon_video, self.stepCount, save_dir="recon_frames")
        # losses
        return latents,interm







class Decoder(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        patch_size,
        image_height = None,
        image_width = None,
        num_latent_tokens = 16,
        encoder_depth = 6,
        decoder_depth = 6,
        time_block_every = 4,
        attn_kwargs: dict = dict(),
        attn_dim_head = 32,
        attn_heads = 8,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        decoder_pos_mlp_depth = 2,
        channels = 3,
        per_image_patch_mask_prob = (0.0, 0.0), # probability of patch masking appears to be per image probabilities drawn uniformly between 0. and 0.9 - if you are a phd student and think i'm mistakened, please open an issue
        lpips_loss_network: Module | None = None,
        lpips_loss_weight = 0.07,
        nd_rotary_kwargs: dict = dict(
            rope_min_freq = 1.,
            rope_max_freq = 10000.,
            rope_p_zero_freqs = 0.
        ),
        num_residual_streams = 1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.final_act = nn.Sigmoid()
        # special tokens

        assert num_latent_tokens >= 1
        self.num_latent_tokens = num_latent_tokens

        self.stepCount=0
        # hyper connections

        hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)

        # mae masking - Kaiming He paper from long ago

        self.per_image_patch_mask_prob = per_image_patch_mask_prob
        self.mask_token = Parameter(randn(dim) * 1e-2)
        encoder_add_decor_aux_loss = True
        self.decorr_aux_loss_weight = 0.03
        decorr_sample_frac = 0.25
        # patch and unpatch
        self.decorr_loss = DecorrelationLoss(decorr_sample_frac, soft_validate_num_sampled = True) if encoder_add_decor_aux_loss else None
        dim_patch = channels * patch_size ** 2


        # self.tokens_to_patch = Sequential(
        #     Linear(dim, dim_patch),
        #     Rearrange('b t h w (p1 p2 c) -> b c t (h p1) (w p2)', p1 = patch_size, p2 = patch_size),
        # )
        self.tokens_to_patch_precalc = Sequential(
            Linear(dim, dim),
            nn.SiLU(),
            Linear(dim, dim),
            nn.SiLU(),
            Linear(dim, dim),
            nn.SiLU()
            )



        self.tokens_to_patch = Sequential(
            Linear(dim, dim_patch),
            Rearrange('b t h w (p1 p2 c) -> b c t (h p1) (w p2)', p1=patch_size, p2=patch_size),
        )
        self.decoder_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)


        # # initialize
        # for layer in self.tok ens_to_patch:
        #     if isinstance(layer, Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)

        # # zero-init the final projection so it starts as a passthrough
        # nn.init.zeros_(self.tokens_t o_patch[4].weight)
        # nn.init.zeros_(self.tokens_t o_patch[4].bias)
        # encoder space / time transformer



        # latents

        self.encoded_to_latents = Sequential(
            ut.LinearNoBias(dim, dim_latent),
            nn.Tanh(),
        )

        self.latents_to_decoder = ut.LinearNoBias(dim_latent, dim)

        # decoder

        self.image_height = image_height
        self.image_width = image_width

        # parameterize the decoder positional embeddings for MAE style training so it can be resolution agnostic

        # self.to_decoder_pos_emb = create_mlp(
        #     dim_in = 2,
        #     dim = dim * 2,
        #     dim_out = dim,
        #     depth = decoder_pos_mlp_depth,
        # )

        # decoder transformer
        # Add after self.to_decoder_pos_emb
        if image_height is not None and image_width is not None:
            num_patches_h = image_height // patch_size
            num_patches_w = image_width // patch_size
            num_patches = num_patches_h * num_patches_w
            
            self.decoder_spatial_pos_embed = Parameter(
                torch.randn(1, 1, num_patches, dim) * 0.02
            )
        else:
            self.decoder_spatial_pos_embed = None
        self.decoder_transformer = trafo.AxialSpaceTimeTransformer(
            dim = dim,
            depth = decoder_depth,
            attn_dim_head = attn_dim_head,
            attn_softclamp_value = attn_softclamp_value,
            time_block_every = time_block_every,
            num_special_spatial_tokens = num_latent_tokens,
            num_residual_streams = num_residual_streams,
            special_attend_only_itself = True,
            final_norm = True
        )

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

        self.has_lpips_loss = lpips_loss_weight > 0.
        self.lpips_loss_weight = lpips_loss_weight

        if self.has_lpips_loss:
            self.lpips = ut.LPIPSLoss(lpips_loss_network)

    @property
    def device(self):
        return self.zero.device

    def muon_parameters(self):
        return [
            *self.encoder_transformer.muon_parameters(),
            *self.decoder_transformer.muon_parameters()
        ]




    def loss_function(        
        self,  
        video, # (b c t h w),
        recon_video, # (b c t h w),
        time_attn_normed_inputs, 
        space_attn_normed_inputs,
        time_attn_normed_inputsDec, 
        space_attn_normed_inputsDec,

        return_all_losses = False
    ):
        recon_loss = F.l1_loss(video, recon_video)

        # print("time_attn_normed_inputs:", time_attn_normed_inputs.shape)
        # print("space_attn_normed_inputs:", space_attn_normed_inputs.shape)
        # print("time_attn_normed_inputsDec:", time_attn_normed_inputsDec.shape)
        # print("space_attn_normed_inputsDec:", space_attn_normed_inputsDec.shape)
        # exit()


        lpips_loss = self.zero
        if ut.exists(self.decorr_loss) and ut.exists(time_attn_normed_inputs):
            time_decorr_loss = self.decorr_loss(to_bnd(time_attn_normed_inputs))

        if ut.exists(self.decorr_loss) and ut.exists(space_attn_normed_inputs):
            space_decorr_loss = self.decorr_loss(to_bnd(space_attn_normed_inputs))

        if ut.exists(self.decorr_loss) and ut.exists(time_attn_normed_inputsDec):
            time_decorr_lossDec = self.decorr_loss(to_bnd(time_attn_normed_inputsDec))

        if ut.exists(self.decorr_loss) and ut.exists(space_attn_normed_inputsDec):
            space_decorr_lossDec = self.decorr_loss(to_bnd(space_attn_normed_inputsDec))

        if self.has_lpips_loss:
            lpips_loss = self.lpips(video, recon_video)

        # losses
        mean_loss = torch.abs(video.mean(dim=[1,2,3]) - recon_video.mean(dim=[1,2,3])).mean()

        total_loss = (
            recon_loss +
            lpips_loss * self.lpips_loss_weight +

            space_decorr_loss * self.decorr_aux_loss_weight +
            time_decorr_loss * self.decorr_aux_loss_weight +
            space_decorr_lossDec * self.decorr_aux_loss_weight*0.5 +
            time_decorr_lossDec * self.decorr_aux_loss_weight*0.5 +
            mean_loss*0.05

        )

        # ff=total_loss.item() # for logging purposes, to get the raw number out
        # gg=recon_loss.item()
        if not return_all_losses:
            return total_loss, mean_loss*0.05

        losses = (recon_loss, lpips_loss, space_decorr_loss, time_decorr_loss)

        return total_loss, TokenizerLosses(*losses)

    def forward(
            
        self,
        latents, # (b t n d)
        height = None,
        width = None,
        inference = False,
    ): # (b c t h w)
        # progress=read_number()
        # divFactor=interpolate(progress,70.0,1.0)        
        if inference:
            self.decoder_transformer.eval()
        else:
            self.decoder_transformer.train()
        

        height = ut.default(height, self.image_height)
        width = ut.default(width, self.image_width)

        assert ut.exists(height) and ut.exists(width), f'image height and width need to be passed in when decoding latents'

        batch, time, device = *latents.shape[:2], latents.device
        flex_attention = None
        use_flex = latents.is_cuda and ut.exists(flex_attention)

        num_patch_height = height // self.patch_size
        num_patch_width = width // self.patch_size
        num_patches = num_patch_height * num_patch_width
        # latents to tokens

        latent_tokens = self.latents_to_decoder(latents)

 
        # Instead of: decoder_pos_tokens = torch.zeros(batch, time, num_patches, d)
        decoder_pos_tokens = repeat(self.mask_token, 'd -> b t n d', b=batch, t=time, n=num_patches)

        # 3. Add the unique positional info
        if self.decoder_spatial_pos_embed is not None:
            decoder_pos_tokens = decoder_pos_tokens + self.decoder_spatial_pos_embed 



        # Concatenate with latent tokens
        tokens, packed_latent_shape = pack((decoder_pos_tokens, latent_tokens), 'b t * d')
        # decoder attention

        tokens, interm  = self.decoder_transformer(tokens, return_intermediates = True)

        # unpack latents

        tokens, latent_tokens = unpack(tokens, packed_latent_shape, 'b t * d')

        # project back to patches
        tokens = rearrange(tokens, 'b t (h w) d -> b t h w d', 
                        h=num_patch_height, w=num_patch_width)

        tokens = self.tokens_to_patch_precalc(tokens)+tokens

        recon_video = self.tokens_to_patch(tokens)

        recon_video = rearrange(recon_video, 'b c t h w -> (b t) c h w')
        recon_video = self.decoder_conv(recon_video)
        recon_video = rearrange(recon_video, '(b t) c h w -> b c t h w', t=time)
        recon_video = recon_video.float() * 8.0
        recon_video = self.final_act(recon_video-4.0)

        return recon_video, interm


    