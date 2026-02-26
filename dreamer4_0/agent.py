from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from . import utils as ut
import torch
from torch import is_tensor
from torch.nn import Module
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
from adam_atan2_pytorch import MuonAdamAtan2
from torch.optim import Optimizer
# from . import pinpadHelper as pp
# from . import pinpadHelper as pp
import torch.nn.utils as nn_utils
from . import tokenizer
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, full, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange
from einops import einsum, rearrange, repeat, reduce, pack, unpack
import numpy as np
# import MCinterface as mci
from torch.utils.checkpoint import checkpoint
import DSfromMP4 as dsmp4
# pinpad DS params
N_TOK_EPISODES = 12
N_BC_EPISODES = 14
EPISODE_LENGTH = 100
DEVICE = 'cuda'
USE_MOTION_PLANNER = True
RAND_LIM = 0.5
USE_MC = True
from hl_gauss_pytorch import HLGaussLoss

from .myLogger import setup_logger

# Set up logging FIRST, before any other imports that might use logging
setup_logger()

import logging
import os
logger = logging.getLogger(__name__)

logger.debug("Agent started")

from dreamer4_0.dynamicsModel import DynamicsWorldModel
# tokenizer params
dim, dim_latent = 1024, 16
patch_size = 16  # Critical change
image_height = 176
image_width = 320
encoder_depth = 12
decoder_depth = 9
attn_heads = 8
time_block_every = 4
num_latent_tokens = 256
returnLength = 11 #seq lenght really
batch_size: int = 1
numTestStep= 899 # training params token
numEpocs = 140
data_from_film = True
useMCds = True
attn_dim_head = 64
attn_softclamp_value = 50.
decoder_pos_mlp_depth = 2
learning_rate: float = 1e-4
weight_decay: float = 0.0
optim_klass: type[torch.optim.Optimizer] = AdamW
max_grad_norm: Optional[float] = 1.0
num_train_steps: int = 10000
from torch.amp import autocast



weight_decayTT = 1e-5
accum_steps = 6
printEvery = 500
#head params
policy_lr = 3e-4
value_lr = 3e-4
weight_decay = 1e-4
tokLr= 6e-5

def save_checkpoint(modelAssembler, optimizer, epoch, losses, out_dir="backupCh"):
    """
    Saves a training checkpoint to out_dir/checkpoint_epoch_{epoch}.pt
    Creates out_dir if it does not exist.
    """
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"checkpoint_epoch.pt")
    torch.save(
        {
            "encoder_state_dict": modelAssembler.encoder.state_dict(),
            "decoder_state_dict": modelAssembler.decoder.state_dict(),

        },
        path,
    )
    print(f"Checkpoint saved -> {path}")
def get_times_from_signal_level(

    signal_levels,
    max_steps,
    align_dims_left_to = None
):
    times = signal_levels.float() / max_steps

    if not ut.exists(align_dims_left_to):
        return times

    return ut.align_dims_left(times, align_dims_left_to)

class modelAssembler(Module):
    def __init__(self):
        super().__init__() 
        self.world_model=None
        self.encoder = None
        self.decoder = None
        self.tokenizer = None
        self.prob_no_shortcut_train = 0.2
        self.num_step_sizes_log2=6
        self.max_steps=2**self.num_step_sizes_log2
        self.pred_orig_latent = True
        self.prevActionTokens=None

        pass


    def forward(self):
        pass

    def encoderBuilder(self):
        print("Building encoder...", num_latent_tokens)
        self.encoder=tokenizer.Encoder(
            dim=dim,
            dim_latent=dim_latent,
            patch_size=patch_size,
            image_height=image_height,
            image_width=image_width,
            encoder_depth=encoder_depth,
            attn_heads=attn_heads,
            time_block_every=time_block_every,
            attn_dim_head=attn_dim_head,
            attn_softclamp_value=attn_softclamp_value,
            num_latent_tokens = num_latent_tokens,
        )
        return self.encoder

    def decoderBuilder(self):
        print("Building decoder...", num_latent_tokens)
        self.decoder=tokenizer.Decoder(
            dim=dim,
            dim_latent=dim_latent,
            patch_size=patch_size,
            image_height=image_height,
            image_width=image_width,
            decoder_depth=decoder_depth,
            attn_heads=attn_heads,
            time_block_every=time_block_every,
            attn_dim_head=attn_dim_head,
            attn_softclamp_value=attn_softclamp_value,
            decoder_pos_mlp_depth=decoder_pos_mlp_depth,
        )
        return self.decoder
        pass

    def selectSignalLevelAndStepSize(
        self,
        batch_size: int=14,
        num_timesteps: int=16,
        k_max: int = 64,
        device: str = 'cuda'
    ):
        """Vectorized version for efficiency."""


        no_shortcut_train = False#ut.sample_prob(self.prob_no_shortcut_train) #focus on SC training for now

        if no_shortcut_train:
            # Standard Flow Matching sampling
            # Step sizes are effectively 1 (2^0)
            step_sizes_log2 = zeros((batch_size,), device = device).long() 
            # Signal levels (t) are sampled uniformly across all possible discrete steps
            signal_levels = randint(0, self.max_steps, (batch_size, num_timesteps), device = device)
        else:
            # Shortcut Training sampling (Consistency)
            # 1. Sample a random step size (power of 2)
            step_sizes_log2 = randint(1, self.num_step_sizes_log2, (batch_size,), device = device)
            num_step_sizes = 2 ** step_sizes_log2

            # 2. Sample signal levels, but discretize them to the sampled step size
            # This ensures 't' and 't + step' land on valid grid points
            signal_levels = randint(0, self.max_steps, (batch_size, num_timesteps), device=device) // num_step_sizes[:, None] * num_step_sizes[:, None]        
            # # Sample step sizes (log2 space for easier indexing)
            # max_log2 = int(np.log2(k_max))
            # step_sizes_log2 = torch.randint(
            #     0, max_log2 + 1, 
            #     (batch_size,), 
            #     device=device
            # )
        add_reward_embed_to_agent_token = False    
        reward_encoder_kwargs: dict = dict()
        self.no_shortcut_train = no_shortcut_train
        self.signal_levels = signal_levels
        self.step_sizes_log2 = step_sizes_log2
        self.reward_encoder = ut.SymExpTwoHot(
            **reward_encoder_kwargs,
            dim_embed = dim,
            learned_embedding = add_reward_embed_to_agent_token
        )        
        return signal_levels, step_sizes_log2, no_shortcut_train


    def loadModel(self):

        checkpoint = torch.load("data/checkpoint4.pt", map_location='cuda')        
        # Debug: print the to_latent_pred structure
        print("\nto_latent_pred keys and shapes in checkpoint:")
        for k, v in checkpoint['dynamics'].items():
            if 'to_latent_pred' in k:
                print(f"  {k}: {v.shape}")

        print("Loading model from checkpoint............................................................")
        checkpoint = torch.load("data/checkpoint4.pt", map_location='cuda')
        
        # Print checkpoint keys for debugging
        # print("\nCheckpoint dynamics keys:", checkpoint['dynamics'].keys())
        
        # Load tokenizer weights
        tokenizer_weights = checkpoint['tokenizer']
        self.encoder.load_state_dict(tokenizer_weights, strict=False)
        self.decoder.load_state_dict(tokenizer_weights, strict=False)
        
        # Load world model dynamics with strict=False
        dynamics_state = checkpoint['dynamics']
        
        # Copy policy_head weights to policy_head_online and policy_head_lagging
        policy_head_keys = [k for k in dynamics_state.keys() if k.startswith('policy_head.')]
        
        for key in policy_head_keys:
            # Create corresponding keys for online and lagging heads
            online_key = key.replace('policy_head.', 'policy_head_online.')
            lagging_key = key.replace('policy_head.', 'policy_head_lagging.')
            
            # Copy the weights
            dynamics_state[online_key] = dynamics_state[key].clone()
            dynamics_state[lagging_key] = dynamics_state[key].clone()
        
        print("Loading model from checkpoint............................................................DONE")
        # Now load the modified state dict
        self.world_model.load_state_dict(dynamics_state, strict=False)
        
        print("Model loaded successfully")
        print(f"Copied {len(policy_head_keys)} policy_head weights to online and lagging heads")


    def initModels(
        self,
        ):
        self.encoder=self.encoderBuilder().cuda()
        self.decoder=self.decoderBuilder().cuda()
        # self.world_model=DynamicsWorldModel(
        #     dim,
        #     dim_latent,
        #     num_latent_tokens=num_latent_tokens,
        #     num_spatial_tokens=num_latent_tokens,

        #     )
        
        if not USE_MC:
            self.loadModel()


        # num_spatial_tokens = 2,        # latents projected to greater number of spatial tokens
        # num_latent_tokens = None,


    def WMmodel(
            self,
            video=None,
            actions=None,
            tasks=None,      
            
            returnActions=False,
            returnRewards=False,
            returnValues=False,
            noisedLatentsOld=None,
            signal_levelsOld=None,
            step_sizes_log2Old=None,
            justReturnPrediction=False,

        ):#no need tasks, actions optionally. may be used for phase 1 and 2
        actions = actions.cuda() if ut.exists(actions) else None

        if(noisedLatentsOld !=None):   
            noisedLatTokens=noisedLatentsOld
            signalLevels=signal_levelsOld
            stepSizesLog2=step_sizes_log2Old
            stepSizesLog2=stepSizesLog2.squeeze(-1)#dirty hack for now
            no_shortcut_train=False
            batch_size=noisedLatTokens.shape[0]
            NumTimeSteps=noisedLatTokens.shape[1]
            latents=None
            times=None
        else:
            print("WM model video shape:", video.shape)
            NumTimeSteps=video.shape[2]
            batch_size=video.shape[0]
            video = video.cuda()
            print("WM model video shape after cuda:", video.shape)
            #encoder expect (b, c, t, h, w)
            latents=self.encoder(video)
            print("WM model latents shape:", latents.shape)
            # latents;  Shape: (b, t, n, d_out) where:
            # b = batch size
            # t = temporal length (number of frames)/sequence length
            # n = number of learned latent tokens (self.latent_tokens.shape[0])
            # d_out = output dimension of self.encoded_to_latents

            signalLevels, stepSizesLog2,no_shortcut_train = self.selectSignalLevelAndStepSize(batch_size=batch_size)
            noisedLatTokens, times = self.noiseLatents(latents,signal_levels=signalLevels,latent_is_noised=False)
        registerTokens=self.registerBuilders(batch_size,NumTimeSteps)               #these things shouldnt be called builder, its models
        print(actions.shape if actions is not None else "No actions provided", "   actions shape in WM model")
        actionTokens=self.embedBuilderActions(actions, NumTimeSteps,batch_size)
        taskTokens=self.embedBuilderTasks(tasks, NumTimeSteps,batch_size)
        #taskTokens; (b, num_agents, dim) 
        # signalLevels, stepSizesLog2, sct = self.selectSignalLevelAndStepSize()
        flowTokens=self.embedBuilderFlow(signalLevels, stepSizesLog2, NumTimeSteps)
        tokenList=[noisedLatTokens, taskTokens, actionTokens, flowTokens, registerTokens]
        for item in tokenList:
            print(item.shape, "   token shape in WM model")

        newLatents,otherStuff=self.world_model(#other is kv cache and some other things tbd later
            latentTokens=noisedLatTokens,
            agent_tokens=taskTokens,
            register_tokens=registerTokens,
            flow_tokens=flowTokens,
            action_tokens = actionTokens, 
            )
        
        if justReturnPrediction:
            xx1 = otherStuff[0]
            xx2 = otherStuff[1]
            print(type(xx1), "   xx1 type in justReturnPrediction")
            print(type(xx2), "   xx2 type in justReturnPrediction")
            print(type(newLatents), "   newLatents type in justReturnPrediction")
            
            # Create tuple explicitly
            result = (newLatents, xx1, xx2)
            print(f"TUPLE LENGTH BEFORE RETURN: {len(result)}")
            print(f"TUPLE TYPE: {type(result)}")
            print(f"TUPLE CONTENTS: {[type(x) for x in result]}")
            
            return result
        encoded_agent_tokens=otherStuff[0]
        # latentTokens,
        # agent_tokens,
        # register_tokens,
        # flow_tokens,
        # action_tokens = None,        

        # newLatents=self.world_model(interleavedTokens)
        rewards=None
        rewardLogprobs=None
        policyEmbed=None
        policyValues=None
        actionLogprobs=None
        valueLogprobs=None
        rewardLogits=None

        if returnRewards:
            rewardLogits=self.rewardHead(encoded_agent_tokens)
            
        if returnActions:
            policyEmbed=self.actionHeadBuilder(encoded_agent_tokens,NumTimeSteps,actions)
            actionsPreds=None#something must be done here
        if returnValues:
            policyValues, valueLogprobs=self.policyValueHeadBuilder(encoded_agent_tokens)

        return newLatents, latents,noisedLatTokens,times, rewardLogits,rewardLogprobs, policyEmbed,  actionLogprobs,policyValues,valueLogprobs

    def WMph1Loss(#sort of works with or without actions. but without actions, the action and reward loss part should be skipped. currently not doing that
        self,
        dataBatch
        ):
        
        self.world_model.train()
        if isinstance(dataBatch["video"], (tuple, list)):
            video = dataBatch["video"][0]
            actions = dataBatch["actions"][0]
            rewards = dataBatch["rewards"][0]
        else:
            video = dataBatch["video"]
            actions = dataBatch["discrete_actions"]
            rewards = dataBatch["rewards"]

        NewLatents,cleanLatents,noisedLatTokens,times,rewardLogits,rewardLogprobs, policyEmbed,  actionLogprobs,_,_=self.WMmodel(
            video=video,
            actions=actions,
            tasks=None,      
            returnActions=True,
            returnRewards=True,
            returnValues=False,
            noisedLatentsOld=None,
            signal_levelsOld=None,
            step_sizes_log2Old=None,
            )
            
        # flowLoss=self.flowLoss(NewLatents,cleanLatents, self.signal_levels, self.step_sizes_log2, self.no_shortcut_train)
        pred = NewLatents
        pred_target = None

        is_x_space = self.pred_orig_latent
        is_v_space_pred = not self.pred_orig_latent

        maybe_shortcut_loss_weight = 1.

        if self.no_shortcut_train:
            print("clean training")

            pred_target = cleanLatents
        else:
            # shortcut training - Frans et al. https://arxiv.org/abs/2410.12557

            # basically a consistency loss where you ensure quantity of two half steps equals one step
            # dreamer then makes it works for x-space with some math
            print("shortcut training")
            step_sizes_log2_minus_one = self.step_sizes_log2 - 1 # which equals d / 2
            half_step_size = 2 ** step_sizes_log2_minus_one

#newLatents, latents,noisedLatTokens,times, rewards,rewardLogprobs, actions,  actionLogprobs,policyValues,valueLogprobs
            NewLatents,cleanLatents,noisedLatTokens,_,_,_,_,_,_,_=self.WMmodel(
                video=None,
                actions=None,
                tasks=None,      
                returnActions=False,
                returnRewards=False,
                returnValues=False,
                noisedLatentsOld=noisedLatTokens,
                signal_levelsOld=self.signal_levels,
                step_sizes_log2Old=step_sizes_log2_minus_one,
                )            # first derive b'
            first_step_pred = NewLatents
            if is_v_space_pred:
                first_step_pred_flow = first_step_pred
            else:
                print(type(self.signal_levels), "   signal levels type in WMph1L oss")
                print(type(noisedLatTokens), "   noised lat tokens type in WM ph1Loss")
                print(type(first_step_pred), "   first step pred type in WMph 1Loss")

                first_times = get_times_from_signal_level(self.signal_levels, self.max_steps, noisedLatTokens)


                first_step_pred_flow = (first_step_pred - noisedLatTokens) / (1. - first_times)

            # take a half step

            half_step_size_align_left = ut.align_dims_left(half_step_size, noisedLatTokens)

            denoised = noisedLatTokens + first_step_pred_flow * (half_step_size_align_left / self.max_steps)

            # get second prediction for b''

            signal_levels_plus_half_step = self.signal_levels + half_step_size[:, None]


            NewLatents,cleanLatents,noisedLatTokens,_,_,_,_,_,_,_=self.WMmodel(
                video=None,
                actions=None,
                tasks=None,      
                returnActions=False,
                returnRewards=False,
                returnValues=False,
                noisedLatentsOld=noisedLatTokens,
                signal_levelsOld=self.signal_levels,
                step_sizes_log2Old=step_sizes_log2_minus_one,
                )            # first derive b'
            second_step_pred = NewLatents

            # second_step_pred = wrapped_get_prediction(denoised, signal_levels_plus_half_step, step_sizes_log2_minus_one)

            if is_v_space_pred:
                second_step_pred_flow = second_step_pred
            else:
                second_times = get_times_from_signal_level(signal_levels_plus_half_step,self.max_steps, denoised)
                second_step_pred_flow = (second_step_pred - denoised) / (1. - second_times)

            # pred target is sg(b' + b'') / 2

            pred_target = (first_step_pred_flow + second_step_pred_flow).detach() / 2

            # need to convert x-space to v-space

            if is_x_space:
                pred = (pred - noisedLatTokens) / (1. - first_times)
                maybe_shortcut_loss_weight = (1. - first_times) ** 2

        # mse loss

        flow_losses = F.mse_loss(pred, pred_target, reduction = 'none')

        flow_losses = flow_losses * maybe_shortcut_loss_weight # handle the (1-t)^2 in eq(7)

        # loss weighting with their ramp function
        def ramp_weight(t):
            return 0.9 * t + 0.1
        
        if ut.exists(ramp_weight):
            loss_weight = ramp_weight(times)
            loss_weight = ut.align_dims_left(loss_weight, flow_losses)

            flow_losses = flow_losses * loss_weight

        # handle variable lengths if needed

        is_var_len = False #ut.exists(lens)

        if is_var_len:

            loss_mask = lens_to_mask(lens, time)
            loss_mask_without_last = loss_mask[:, :-1]

            flow_loss = flow_losses[loss_mask].mean()

        else:
            flow_loss = flow_losses.mean()
        actionLoss= self.calcActionLoss(actions,policyEmbed)
        rewardLoss= self.calcRewardLoss(rewards,rewardLogits)
        totalLoss=flow_loss+actionLoss.sum()+rewardLoss.sum()
        return totalLoss

    def calcActionLoss(
        self,
        discrete_actions,
        policy_embed,
    ):

            # constitute multi token prediction targets
            discrete_actions = discrete_actions.cuda() if ut.exists(discrete_actions) else None
            discrete_action_targets = continuous_action_targets = None

            if ut.exists(discrete_actions):
                discrete_action_targets, discrete_mask = ut.create_multi_token_prediction_targets(discrete_actions, self.world_model.multi_token_pred_len)
                discrete_action_targets = rearrange(discrete_action_targets, 'b t mtp ... -> mtp b t ...')
                discrete_mask = rearrange(discrete_mask, 'b t mtp -> mtp b t')
                discrete_action_targets=discrete_action_targets.to(policy_embed.device)

            print("policy embed device:", policy_embed.device)
            print("world model action embedder device:", self.world_model.action_embedder.device)
            print("discrete action targets device:", discrete_action_targets.device if ut.exists(discrete_action_targets) else "N/A")
            print("discrete_action device:", discrete_actions.device if ut.exists(discrete_actions) else "N/A")

            discrete_log_probs = self.world_model.action_embedder.log_probs(
                policy_embed,
                discrete_targets = discrete_action_targets if ut.exists(discrete_actions) else None#,
                # continuous_targets = continuous_action_targets[:, :-1] if False else None
            )

            if ut.exists(discrete_log_probs):
                discrete_log_probs = discrete_log_probs.masked_fill(~discrete_mask[..., None], 0.)

                if False:#is_var_len:
                    discrete_action_losses = rearrange(-discrete_log_probs, 'mtp b t na -> b t na mtp')
                    discrete_action_loss = reduce(discrete_action_losses[loss_mask_without_last], '... mtp -> mtp', 'mean')
                else:
                    discrete_action_loss = reduce(-discrete_log_probs, 'mtp b t na -> mtp', 'mean')

            return discrete_action_loss


    def calcRewardLoss(
        self,
        rewards,
        rewardLogits

    ):
        # reward loss

        two_hot_encoding = self.reward_encoder(rewards)

        # reward_loss = self.zero

        # if rewards.ndim == 2: # (b t)
        #     encoded_agent_tokens = reduce(encoded_agent_tokens, 'b t g d -> b t d', 'mean')

        # reward_pred = self.to_rewa rd_pred(encoded_agent_tokens[:, :-1])

        # reward_pred = rearrange(reward_pred, 'mtp b t l -> b l t mtp')

        reward_targets, reward_loss_mask = ut.create_multi_token_prediction_targets(two_hot_encoding[:, :-1], self.world_model.multi_token_pred_len)
        reward_loss_mask = reward_loss_mask.to(rewardLogits.device)
        reward_targets = rearrange(reward_targets, 'b t mtp l -> b l t mtp').cuda()

        reward_losses = F.cross_entropy(rewardLogits, reward_targets, reduction = 'none')

        reward_losses = reward_losses.masked_fill(~reward_loss_mask, 0.)

        if False:#is_var_len:
            reward_loss = reward_losses[loss_mask_without_last].mean(dim = 0)
        else:
            reward_loss = reduce(reward_losses, '... mtp -> mtp', 'mean') # they sum across the prediction steps (mtp dimension) - eq(9)

        return reward_loss


    def embedBuilderActions(
        self,
        actions,
        NumTimeSteps,        
        batch
        ):#embed tasks, actions etc

        assert self.world_model.action_embedder.has_actions
        assert self.world_model.num_agents == 1, 'only one agent allowed for now'
        agent_tokens = repeat(self.world_model.agent_learned_embed, '... d -> b ... d', b = batch)
        agent_tokens = repeat(agent_tokens, 'b ... d -> b t ... d', t = NumTimeSteps)
        print(agent_tokens.shape, "   agent tokens shape in embedBuilderActions")
        if ut.exists(actions):

            # actions = actions.to(agent_tokens.cuda)

            # if actions come as float 0.0/1.0, make them valid embedding indices
            actions = actions.round().clamp(0, 1).to(torch.long)

            # (optional but extremely helpful during debugging)
            assert actions.dtype == torch.long
            assert actions.min().item() >= 0
            assert actions.max().item() <= 1
            assert actions.shape[-1] == 17, actions.shape


            action_tokens = self.world_model.action_embedder(discrete_actions = actions)
            action_tokens = action_tokens.sum(dim=-2, keepdim=True)
            #action_tokens = rearrange(action_tokens, 'b t d -> b t 1 d')                 #this is number of action being treated as number of agents. both are 1 mostlly so ok for now
            logger.debug(f'action tokens shape before cat: {action_tokens.shape}')    
            if self.prevActionTokens is not None:
                action_tokens = torch.cat((self.prevActionTokens, action_tokens[:, :-1]), dim=1)
                
        else:
            action_tokens = torch.zeros_like(agent_tokens[:, :, 0:1])#first frame should have actions to. to be fixed later
            self.prevActionTokens= action_tokens.cuda()
        logger.debug(f'action tokens shape before to device: {action_tokens.shape}')    
        action_tokens = action_tokens.to(agent_tokens.device)
        print(agent_tokens.shape, "   agent tokens shape before pad")
        print(action_tokens.shape, "   action tokens shape before pad")

        # handle first timestep not having an associated past action

        if action_tokens.shape[1] == (NumTimeSteps - 1):
            action_tokens = ut.pad_at_dim(action_tokens, (1, 0), value = 0. , dim = 1)
        print(action_tokens.shape, "   action tokens shape after pad")
        action_tokens = ut.add('1 d, b t 1 d', self.world_model.action_learned_embed, action_tokens)
        print(action_tokens.shape)


        return action_tokens


        #mbed flow
            # determine signal + step size embed for their diffusion forcing + shortcut


    def embedBuilderTasks(
        self,
        tasks,
        NumTimeSteps,
        numBatches
        ):#embed tasks, actions etc


        #embed tasks
        agent_tokens = repeat(self.world_model.agent_learned_embed, '... d -> b ... d', b = numBatches)

        if ut.exists(tasks):
            assert self.num_tasks > 0

            task_embeds = self.world_model.task_embed(tasks)
            agent_tokens = ut.add('b ... d, b d', agent_tokens, task_embeds)        

        return repeat(agent_tokens, 'b ... d -> b t ... d', t = NumTimeSteps)


    def embedBuilderFlow(
        self,

        signal_levels,
        step_sizes_log2,
        NumTimeSteps
        ):#embed tasks, actions etc
        print(signal_levels.device, "   signal levels device in embedBuilderF low")
        print(self.world_model.signal_levels_embed.weight.device, "   signal levels embed device in embedBuil derFlow")
        print("")
        signal_embed = self.world_model.signal_levels_embed(signal_levels)
        step_size_embed = self.world_model.step_size_embed(step_sizes_log2)
        step_size_embed = repeat(step_size_embed, 'b ... -> b t ...', t = NumTimeSteps)#this cant be right, should have different step size per time step
        if step_size_embed.dim() > 3 and step_size_embed.size(2) == 1:
            step_size_embed = step_size_embed.squeeze(2)    
        print(signal_embed.shape, "   signal embed shape in embedBuil derFlow")
        print(step_size_embed.shape, "   step size embed shape in embedBui lderFlow")    
        flow_token = cat((signal_embed, step_size_embed), dim = -1)
        flow_token = rearrange(flow_token, 'b t d -> b t d')

        return flow_token

    def registerBuilders(self,batch, time
    ):
        registers = repeat(self.world_model.register_tokens, 's d -> b t s d', b = batch, t = time)#i assume s is number of registers
        return registers

    def noiseLatents(self, latents, signal_levels, latent_is_noised):
        times = get_times_from_signal_level(signal_levels,self.max_steps)
        print(latents.shape, "   latents shape in noiseLatents")
        if True:
            # get the noise

            noise = randn_like(latents)
            aligned_times = ut.align_dims_left(times, latents)
            print(aligned_times.device)
            print(latents.device)
            # noise from 0 as noise to 1 as data

            noised_latents = noise.lerp(latents, aligned_times)
        return noised_latents, times

    def interleaveTokens(
        self,
        latents,
        task_tokens,
        action_tokens,
        flow_tokens,
        register_tokens
        ):
        # interleave all tokens
        pass
        

    def rewardHead(self, latents): 



        encoded_agent_tokens = reduce(latents, 'b t g d -> b t d', 'mean')

        reward_pred = self.world_model.to_reward_pred(encoded_agent_tokens[:, :-1])

        reward_pred = rearrange(reward_pred, 'mtp b t l -> b l t mtp')
        return reward_pred


    def actionHeadBuilder(self,agent_tokens,time,discrete_actions):
        # maybe autoregressive action loss

        # discrete_action_loss = self.zero

        add_autoregressive_action_loss=True
        if (
            True
            # add_autoregressive_action_loss and
            # time > 1,
            # (exists(discrete_actions) or exists(continuous_actions))
        ):
            assert self.world_model.action_embedder.has_actions

            # handle actions having time vs time - 1 length
            # remove the first action if it is equal to time (as it would come from some agent token in the past)

            if ut.exists(discrete_actions) and discrete_actions.shape[1] == time:
                discrete_actions = discrete_actions[:, 1:]


            # only for 1 agent

            agent_tokens = rearrange(agent_tokens, 'b t 1 d -> b t d')
            policy_embed = self.world_model.policy_head_lagging(agent_tokens)

            return policy_embed

        def policyValueHeadBuilder(
            self,
        ):
            policy_embed = self.policy_head(agent_embeds)

            log_probs, entropies = self.action_embedder.log_probs(policy_embed, pred_head_index = 0, discrete_targets = discrete_actions, continuous_targets = continuous_actions, return_entropies = True)

                #predict actions
                # return  log_probs and-or actions       

#single decoder for video generation for test
def collate_fn(batch):
    """Collate function that handles None values in dicts"""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # Check if all values are None
        if all(v is None for v in values):
            collated[key] = None
        # Check if some are None (mixed batch - shouldn't happen but handle it)
        elif any(v is None for v in values):
            raise ValueError(f"Mixed None/non-None values for key '{key}' in batch")
        # All values are present - collate normally
        else:
            collated[key] = torch.stack(values)
    
    return collated


def crop_or_pad_to_patch_size(video, patch_size=16):

    B, C, T, H, W = video.shape
    target_H = (H // patch_size) * patch_size
    target_W = (W // patch_size) * patch_size

    if target_H == 0 or target_W == 0:
        return video  # or pad instead

    h_start = H - target_H          # crop TOP, keep bottom
    w_start = (W - target_W) // 2   # center crop width
    return video[:, :, :, h_start:h_start+target_H, w_start:w_start+target_W]

def read_status(path="status.txt"):
    try:
        with open(path, "r") as f:
            first_char = f.read(1)  # read only first character
    except (FileNotFoundError, OSError):
        return 0

    if first_char == "1":
        return 1
    elif first_char == "2":
        return 2
    else:
        return 0


class DreamerAgent():
    def __init__(self):
        self.batch_size=16
        super().__init__()
        self.modelAssembler=modelAssembler()
        # default optimizers (user can override by calling learn with their own)
        # self.policy_optim = Adam(self.world_model.policy_head_parameters(), lr=policy_lr, weight_decay=weight_decay)
        # self.value_optim = Adam(self.world_model.value_head_parameters(), lr=value_lr, weight_decay=weight_decay)
        self.tok_ds = None
        self.bc_ds = None
        self.latent_shape = (4, 256)  # Example latent shape (num_latent_tokens, dim_latent)
        reward_range = (0.0, 1000.0)
        critic_pred_num_bins = 250
        self.optimTokenizer = None
        self.optimWM = None
        self.optimPolicy = None
        self.optimValue = None
        self.optimReward = None


        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )


    def initOptimizers(
        self,
        learning_rateTokenizer=learning_rate,
        weight_decayTokenizer=weight_decayTT,
        optim_klassTokenizer=optim_klass,
        learning_rateWM=learning_rate,
        weight_decayWM=weight_decayTT,
        optim_klassWM=optim_klass,
        learning_ratePolicy=policy_lr,
        weight_decayPolicy=weight_decay,
        optim_klassPolicy=optim_klass,
        learning_rateValue=value_lr,        
        weight_decayValue=weight_decay,
        optim_klassValue=optim_klass,
        learning_rateReward=learning_rate,
        weight_decayReward=weight_decay,
        optim_klassReward=optim_klass,
        ):
        self.optimTokenizer = optim_klassTokenizer(
            [
                {'params': self.modelAssembler.encoder.parameters()},
                {'params': self.modelAssembler.decoder.parameters()}
            ],
            lr=tokLr,
            weight_decay=weight_decayTokenizer
        )
        # self.optimWM = optim_klassWM(
        #     self.modelAssembler.world_model.parameters(),
        #     lr = learning_rateWM,
        #     weight_decay = weight_decayWM
        # )


        # self.optimPolicy = optim_klassPolicy(
        #     self.modelAssembler.world_model.policy_head_online_parameters(),
        #     lr = learning_ratePolicy,
        #     weight_decay = weight_decayPolicy
        # )
        # self.optimValue = optim_klassValue(
        #     self.modelAssembler.world_model.value_head_parameters(),
        #     lr = learning_rateValue,
        #     weight_decay = weight_decayValue
        # )
        # self.optimReward = optim_klassReward(
        #     self.modelAssembler.world_model.reward_pred_parameters(),
        #     lr = learning_rateReward,
        #     weight_decay = weight_decayReward
        # )

    # Example usage with batched sequences
    def compute_lambda_returns_batched(self,rewards, values, continues, gamma=0.997, lambda_=0.95):
        """
        Batched version for [B, T] shaped tensors.
        
        Args:
            rewards: torch.Tensor [B, T]
            values: torch.Tensor [B, T]
            continues: torch.Tensor [B, T]
            gamma: float
            lambda_: float
        
        Returns:
            lambda_returns: torch.Tensor [B, T]
        """
        B, T = rewards.shape
        lambda_returns = torch.zeros_like(rewards)
        
        # Bootstrap from final values
        lambda_returns[:, -1] = values[:, -1]
        
        # Backward pass
        for t in reversed(range(T - 1)):
            lambda_returns[:, t] = (
                rewards[:, t] + 
                gamma * continues[:, t] * (
                    (1 - lambda_) * values[:, t] + 
                    lambda_ * lambda_returns[:, t + 1]
                )
            )
        
        return lambda_returns


    # Then compute advantages for PMPO
    def compute_advantages(self,lambda_returns, values):
        """
        Compute advantages: A_t = R_λ_t - v_t
        
        Args:
            lambda_returns: torch.Tensor [B, T]
            values: torch.Tensor [B, T]
        
        Returns:
            advantages: torch.Tensor [B, T]
        """
        return lambda_returns - values



        
    def MakeDatasets(self):
        """
        Collect on-policy experience from a real environment using the world model's helper.
        """
        
        if useMCds:
            if data_from_film:

                self.bc_ds = dsmp4.build_datasets_from_mp4(

                    return_length=returnLength,

                   
                    resize=(image_width, image_height)  # (width, height)
                )                
                # self.tok_ds, self.bc_ds =dsmp4.build_datasets_from_mp4(
                #     mp4_paths=passmp4Paths,
                #     episode_length=None,  # or some number
                #     return_length=16,
                #     tok_window=16,
                #     tok_stride=10,
                # )
            else:
                self.tok_ds, self.bc_ds = mci.build_datasets_from_npz(

                    npz_paths=["/home/jan/pyPj/dreamer4Dirk/dreamer4-nbs/runs/manual_basalt_findcave/episodes/ep_00000.npz", 
                            "/home/jan/pyPj/dreamer4Dirk/dreamer4-nbs/runs/manual_basalt_findcave/episodes/ep_00001.npz", 
                            "/home/jan/pyPj/dreamer4Dirk/dreamer4-nbs/runs/manual_basalt_findcave/episodes/ep_00002.npz"],  # 3 episodes from 3 files
                    episode_length=None,  # or some number
                    return_length=16,
                    tok_window=16,
                    tok_stride=10,
                )
            print("Building datasets for tokenizer and BC trainer...")
        else:
            self.tok_ds, self.bc_ds, env, _ = pp.build_pinpad_datasets_for_trainers(
                n_tok_episodes=N_TOK_EPISODES,
                n_bc_episodes=N_BC_EPISODES,
                episode_length=EPISODE_LENGTH,
                device=DEVICE,
                use_motion_planner=USE_MOTION_PLANNER,
                randLim=RAND_LIM
            )    
        print("")    



    def trainTokenizer(
        self,
        num_test_steps: int = 30,
        batch_size: int = batch_size,
    ):
        #
        print("Training tokenizer ")
        super().__init__()
        resp = input("Do you want to load encoder/decoder weights? [y/N]: ").strip().lower()
        if resp in ("y", "yes"):
            self.modelAssembler.encoder.load_state_dict(
                torch.load("encoder.pt", map_location="cpu"),
                strict=False
            )
            # state_dict = torch.load("decoder.pt", map_location="cpu")
            # # remove mismatched keys
            # for key in ["tokens_to_patch.0.weight", "tokens_to_patch.0.bias"]:
            #     state_dict.pop(key, None)
            # self.modelAssembler.decoder.load_state_dict(state_dict, strict=False)            

            self.modelAssembler.decoder.load_state_dict(
                torch.load("decoder.pt", map_location="cpu"),
                strict=False
            )
            print("Weights loaded.")
        else:
            print("Starting from random initialization.")        
        self.num_test_steps = num_test_steps
        self.dataset = self.tok_ds
        # Usage:
        # sampler = ut.RepeatEpisodeSampler(self.bc_ds, batch_size=batch_size, samples_per_episode=8)

        self.train_dataloader = DataLoader(self.bc_ds, batch_size = batch_size, drop_last = True, shuffle = False)#


        self.batch_size = batch_size
        iter_train_dl = ut.cycle(self.train_dataloader)
        self.optimTokenizer.zero_grad()
        losses = []
        runSts=0
        meanLossAcc=[0.0,0.0]
        for j in range(numEpocs):#epochs
            if runSts>0:
                break   
            if (j + 1) % 10 == 0:
                save_checkpoint(
                    modelAssembler=self.modelAssembler,
                    optimizer=self.optimTokenizer,
                    epoch=j + 1,
                    losses=losses,
                    out_dir="backupCh",
                )                     
            self.bc_ds.initNewEpoch()#shuffle episodes for next epoch, if desired. not needed if shuffle = True in dataloader
            print(f"Epoch {j+1} / {numEpocs}")
            for i in range(self.num_test_steps):
                runSts=read_status()
                if runSts>0:
                    print(f"Run status {runSts} detected, stopping training loop.")
                    break
                # robust to both TensorDataset and bare-tensor datasets
                batch = next(iter_train_dl)
                batch = batch["video"] if isinstance(batch, dict) else batch
                video = batch[0] if isinstance(batch, (tuple, list)) else batch
                #video to gpu
                video = video.cuda()
                video = crop_or_pad_to_patch_size(video, patch_size=patch_size)
                
                # Zero gradients at the start of accumulation
                if i % accum_steps == 0:
                    self.optimTokenizer.zero_grad(set_to_none=True)
                hhh=(i % printEvery == 0)
                # print(hhh)
                # === MIXED PRECISION FORWARD (NEW API) ===
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Forward pass in BF16
                    z, (_, time_attn_normed_inputs, space_attn_normed_inputs, _) = self.modelAssembler.encoder(video, inference=False)

                    vid, (_, time_attn_normed_inputs2, space_attn_normed_inputs2, _) = checkpoint(self.modelAssembler.decoder,z, height=176, width=320, inference=False, use_reentrant=False)
                    
                    # Loss computation in BF16
                    loss, meanloss = self.modelAssembler.decoder.loss_function(video,vid,time_attn_normed_inputs, space_attn_normed_inputs ,time_attn_normed_inputs2, space_attn_normed_inputs2  )
                meanLossAcc[0]+=meanloss.item()
                meanLossAcc[1]+=1

                # === BACKWARD IN FP32 ===
                (loss / accum_steps).backward()
                
                # Step optimizer when accumulation is done
                do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == self.num_test_steps)
                if do_step:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.modelAssembler.parameters(), 
                        max_norm=1.0
                    )
                    
                    self.optimTokenizer.step()
                    self.optimTokenizer.zero_grad(set_to_none=True)
                
                # Logging
                if i % printEvery == 0:
                    ut.save_random_recon_comparison(video, vid, step=j)
                    print("Saved reconstruction comparison video for step", i)
    
                    print(f"Step {i}: Loss = {loss.item():.4f}")
                    print(
                        "Max reserved:",
                        torch.cuda.max_memory_reserved() / 1024**2,
                        "MB"
                    )            
                    print("meanloss avg: ",meanLossAcc[0]/meanLossAcc[1])
                    meanLossAcc[0]=0.0
                    meanLossAcc[1]=0.0
                      

                        
                losses.append(loss.detach().item())
        #resp = input("Do you want to save encoder/decoder weights? [y/N]: ").strip().lower()
        if runSts < 2:# resp in ("y", "yes"):
            torch.save(self.modelAssembler.encoder.state_dict(), "encoder.pt")
            torch.save(self.modelAssembler.decoder.state_dict(), "decoder.pt")
            print("Weights saved.")
        else:
            print("Weights not saved.")

        return losses


    def test(#
        self,
    ):
        #bc_ds.samples[i]["video"] is (C,T,H,W) on CPU.
        # C – number of channels (e.g. 3 for RGB)
        # T – episode_length

        print((len(self.bc_ds)))
        batch_size = min(self.batch_size, len(self.bc_ds))
        # self.dataset = self.bc_ds
        self.train_dataloader = DataLoader(self.bc_ds, batch_size = batch_size, drop_last = True, shuffle = True)
        iter_train_dl = ut.cycle(self.train_dataloader)# (b c t h w)
        batch = next(iter_train_dl)
        print(len(batch), "batch len")

        self.modelAssembler.initModels()
        self.initOptimizers()
        self.optimWM.zero_grad()
        #add epochs later
        for batch_idx, batch in enumerate(self.train_dataloader):        
#        print(video.shape,"   vid shape")
            video = batch["video"][0]#test, to be deleted later
            loss=self.modelAssembler.WMph1Loss(batch)
            loss.backward()
            self.optimWM.step()
            self.optimWM.zero_grad()



    def dNoise(
        self,
        num_steps: int,
        step_size: int,
        batch_size: int,
        take_extra_step: bool,
        curr_time_steps: int,
        latents: Tensor,
        past_latents_context_noise: Tensor,
        context_signal_noise: float,
        noised_latent: Tensor,
        tasks: Optional[Tensor],

        decoded_discrete_actions: Optional[Tensor],

        time_kv_cache: Optional[Tensor],
        use_time_kv_cache: bool,

    ):
        """
        Add Gaussian noise to a tensor.
        """

        for step in range(num_steps + int(take_extra_step)):
            logger.debug(f'Diffusion step {step + 1} / {num_steps},  step no:{self.genSteps}')

            is_last_step = (step + 1) == num_steps

            signal_levels = full((batch_size, 1), step * step_size, dtype = torch.long).cuda()  # (b 1)

            # noising past latent context

            noised_context = latents.lerp(past_latents_context_noise, context_signal_noise) # the paragraph after eq (8)

            noised_latent_with_context, pack_context_shape = pack([noised_context, noised_latent], 'b * s d')# * is n

            # proper signal levels

            signal_levels_with_context = F.pad(signal_levels, (curr_time_steps, 0), value = self.modelAssembler.max_steps - 1)
            stepSizeL2 = 4 * torch.ones_like(signal_levels_with_context)
            stepSizeL2 = stepSizeL2[:,0:1]#temporary fix. need to revisit later, give different step sizes to different time steps
            raw_result= self.modelAssembler.WMmodel(


                video=None,
                actions=decoded_discrete_actions,
                tasks=None,      
                
                returnActions=False,
                returnRewards=False,
                returnValues=False,
                noisedLatentsOld=noised_latent_with_context,
                signal_levelsOld=signal_levels_with_context,
                step_sizes_log2Old=stepSizeL2,
                justReturnPrediction=True
            )
            print("----------------------------------------")

            print(f"========== CALLER SIDE ==========")
            print(f"RECEIVED TYPE: {type(raw_result)}")
            if isinstance(raw_result, tuple):
                print(f"RECEIVED TUPLE LENGTH: {len(raw_result)}")
                print(f"TUPLE CONTENTS: {[type(x) for x in raw_result]}")
            else:
                print(f"NOT A TUPLE! Got: {raw_result}")

            st1, st2, st3 = raw_result

            pred=st1
            agent_embed=st2
            next_time_kv_cache=st3



            if use_time_kv_cache and is_last_step:
                time_kv_cache = next_time_kv_cache

            # early break if taking an extra step for agent embedding off cleaned latents for decoding

            if take_extra_step and is_last_step:
                break

            # maybe proprio



            # unpack pred

            _, pred = unpack(pred, pack_context_shape, 'b * s d')

# out, pack_info = pack([t1, t2], 'b n s d')
# t1_rec, t2_rec = unpack(out, pack_info, 'b n s d')
            # derive flow, based on whether in x-space or not

            def denoise_step(pred, noised, signal_levels):
                if self.modelAssembler.pred_orig_latent:
                    times = get_times_from_signal_level(signal_levels,self.modelAssembler.max_steps)
                    aligned_times = ut.align_dims_left(times, noised)

                    flow = (pred - noised) / (1. - aligned_times)
                else:
                    flow = pred

                return flow * (step_size / self.modelAssembler.max_steps)

            # denoise

            noised_latent += denoise_step(pred, noised_latent, signal_levels)

        denoised_latent = noised_latent
        return denoised_latent, past_latents_context_noise, agent_embed, time_kv_cache


    def generateTrajectory(


        self,
        time_steps,
        num_steps = 4,
        batch_size = 2,
        agent_index = 0,
        tasks: int | Tensor | None = None,
        latent_gene_ids = None,
        image_height = None,
        image_width = None,
        return_decoded_video = None,
        context_signal_noise = 0.1,       # they do a noising of the past, this was from an old diffusion world modeling paper from EPFL iirc
        time_kv_cache: Tensor | None = None,
        use_time_kv_cache = True,
        return_rewards_per_frame = False,
        return_agent_actions = False,
        return_log_probs_and_values = False,
        return_for_policy_optimization = False,
        return_time_kv_cache = False,
        store_agent_embed = True,
        store_old_action_unembeds = True

    ): # (b t n d) | (b c t h w)

        # handy flag for returning generations for rl
        actionList = [None]
        print((len(self.bc_ds)))
        batch_size = 1#min(self.batch_size, len(self.bc_ds))
        # self.dataset = self.bc_ds
        self.train_dataloader = DataLoader(self.bc_ds, batch_size = batch_size, drop_last = True, shuffle = True)
        iter_train_dl = ut.cycle(self.train_dataloader)# (b c t h w)         
        batchInitData = next(iter_train_dl)
        vids=batchInitData["video"]
        vids=vids.cuda()
        print(len(batchInitData), "batch len")
        
        latents=self.modelAssembler.encoder(vids).cuda()


        latents = latents[0:1, 0:1, :, :]
        if return_for_policy_optimization:
            return_agent_actions |= True
            return_log_probs_and_values |= True
            return_rewards_per_frame |= True

        # more variables

        #inference mode
        self.modelAssembler.eval()

        # validation

        assert ut.log2(num_steps).is_integer(), f'number of steps {num_steps} must be a power of 2'
        assert 0 < num_steps <= self.modelAssembler.max_steps, f'number of steps {num_steps} must be between 0 and {self.modelAssembler.max_steps}'

        if isinstance(tasks, int):
            tasks = full((batch_size,), tasks, device = self.device)

        assert not ut.exists(tasks) or tasks.shape[0] == batch_size

        # get state latent shape

        latent_shape = self.latent_shape

        # derive step size

        step_size = self.modelAssembler.max_steps // num_steps

        # denoising
        # teacher forcing to start with



        past_latents_context_noise = latents.clone()

        # maybe internal state



        # maybe return actions

        return_agent_actions |= return_log_probs_and_values

        decoded_discrete_actions = None


        # policy optimization related

        decoded_discrete_log_probs = None
        decoded_discrete_logits = None

        decoded_values = None

        # maybe store agent embed

        acc_agent_embed = None

        # maybe store old actions for kl

        acc_policy_embed = None

        # maybe return rewards

        decoded_rewards = None
        if return_rewards_per_frame:
            decoded_rewards = empty((batch_size, 0),  dtype = torch.float32).cuda()

        # while all the frames of the video (per latent) is not generated
        self.genSteps=0
        while latents.shape[1] < time_steps:
            self.genSteps+=1
            curr_time_steps = latents.shape[1]

            # determine whether to take an extra step if
            # (1) using time kv cache
            # (2) decoding anything off agent embedding (rewards, actions, etc)

            take_extra_step = (
                use_time_kv_cache or
                return_rewards_per_frame or
                store_agent_embed or
                return_agent_actions
            )

            # prepare noised latent / proprio inputs

            noised_latent = randn((batch_size, 1,  *latent_shape)).cuda()

            noised_proprio = None


            # denoise
            denoised_latent, past_latents_context_noise, agent_embed, time_kv_cache = self.dNoise(
                num_steps = num_steps,
                step_size = step_size,
                batch_size = batch_size,
                take_extra_step = take_extra_step,
                curr_time_steps = curr_time_steps,
                latents = latents,
                past_latents_context_noise = past_latents_context_noise,
                context_signal_noise = context_signal_noise,
                noised_latent = noised_latent,

                tasks = tasks,

                decoded_discrete_actions = decoded_discrete_actions,

                time_kv_cache = time_kv_cache,
                use_time_kv_cache = use_time_kv_cache,
            )



            # take care of the rewards by predicting on the agent token embedding on the last denoising step
            one_agent_embed = agent_embed[:, -1:, agent_index] # agent_embed dims ; (b t n d), one_agent_embed dims ; (b 1 d)


            # maybe store agent embed

            if store_agent_embed:
                acc_agent_embed = ut.safe_cat((acc_agent_embed, one_agent_embed), dim = 1)

            # decode the agent actions if needed

            if return_agent_actions:
                assert self.modelAssembler.world_model.action_embedder.has_actions


                policy_embed = self.modelAssembler.world_model.policy_head_lagging(one_agent_embed)



                # sample actions

                sampled_discrete_actions = self.modelAssembler.world_model.action_embedder.sample(policy_embed, pred_head_index = 0, squeeze = True)

                decoded_discrete_actions = ut.safe_cat((decoded_discrete_actions, sampled_discrete_actions), dim = 1)
                actionList.append(sampled_discrete_actions)




            # concat the denoised latent

            latents = cat((latents, denoised_latent), dim = 1)

            # add new fixed context noise for the temporal consistency

            past_latents_context_noise = cat((past_latents_context_noise, randn_like(denoised_latent)), dim = 1)


        # restore state
        self.modelAssembler.train()


        # returning video

        has_tokenizer = ut.exists(self.modelAssembler.encoder) and ut.exists(self.modelAssembler.decoder) 
        return_decoded_video = ut.default(return_decoded_video, has_tokenizer)

        video = None

        if return_decoded_video:

            latents_for_video = rearrange(latents, 'b t n d -> b t n d')
            latents_for_video, unpack_view = ut.pack_one(latents_for_video, '* t n d')

            video = self.modelAssembler.decoder(
                latents_for_video,
                height = image_height,
                width = image_width
            )

            video = unpack_view(video, '* t c vh vw')





        if not ut.has_at_least_one(return_rewards_per_frame, return_agent_actions, False):
            out = video if return_decoded_video else latents

            if not return_time_kv_cache:
                return out

            return out, time_kv_cache

        # returning agent actions, rewards, and log probs + values for policy optimization

        batch, device = latents.shape[0], latents.device
        experience_lens = full((batch,), time_steps, device = device)
        # doit = ut.exists(acc_policy_embed) and store_old_action_unembeds
        # tenstmp = self.modelAssembler.world_model.action_embedder.unembed(acc_policy_embed, pred_head_index = 0)
        gen = ut.Experience(
            latents = latents,
            video = video,
            
            agent_embed = acc_agent_embed if store_agent_embed else None,

            step_size = step_size,
            agent_index = agent_index,
            lens = experience_lens,
            is_from_world_model = True
        )

        if return_rewards_per_frame:
            gen.rewards = decoded_rewards

        if return_agent_actions:
            gen.actions = (decoded_discrete_actions)

        if return_log_probs_and_values:
            gen.log_probs = (decoded_discrete_log_probs)

            gen.values = decoded_values
            gen.logits = decoded_discrete_logits

        if not return_time_kv_cache:
            return gen

        return gen, time_kv_cache



    def headTrainer(
        self,
        experience: ut.Experience,
        
    ):
        dones=None#to be fixed later
        policy_optim: Optimizer | None = None,
        value_optim: Optimizer | None = None,        
        # optimizer=self.modelAssembler.head_optimizer
        latents = experience.latents
        actions = experience.actions
        old_log_probs = experience.log_probs #action log probs
        old_values = experience.values
        rewards_t = experience.rewards
        agent_embeds = experience.agent_embed
        old_action_unembeds = experience.old_action_unembeds

        step_size = experience.step_size
        
        #loop thru agent embeds
        decoded_rewards = None
        decoded_discrete_actions = None
        decoded_discrete_log_probs = None
        decoded_values = None
        decoded_discrete_logits_online = None
        decoded_discrete_logits_lagging = None
        value_logits_all = []
        actionList = []


        # Optimizers belong here
        self.policy_optim = torch.optim.Adam(
            self.modelAssembler.world_model.policy_head_online.parameters(),
            lr=3e-4
        )
        self.value_optim = torch.optim.Adam(
            self.modelAssembler.world_model.value_head.parameters(),
            lr=3e-4
        )


        for t in range(agent_embeds.shape[1]):
            one_agent_embed = agent_embeds[:, t:t+1] # agent 


            reward_logits = (
                    self.modelAssembler.world_model
                    .to_reward_pred(one_agent_embed)[0]
                )
            pred_reward = self.modelAssembler.world_model.reward_encoder.bins_to_scalar_value(reward_logits, normalize = True)

            value_bins = self.modelAssembler.world_model.value_head(one_agent_embed)
            value_logits_all.append(value_bins)
            if decoded_rewards is None:
                decoded_rewards = pred_reward
            else:
                decoded_rewards = cat((decoded_rewards, pred_reward), dim = 1)
            
            assert self.modelAssembler.world_model.action_embedder.has_actions


            policy_embed_online = self.modelAssembler.world_model.policy_head_online(one_agent_embed)
            policy_embed_lagging = self.modelAssembler.world_model.policy_head_lagging(one_agent_embed)

            # maybe store old actions






            value_bins = self.modelAssembler.world_model.value_head(one_agent_embed)
            values = self.modelAssembler.world_model.reward_encoder.bins_to_scalar_value(value_bins)

            decoded_values = ut.safe_cat((decoded_values, values), dim = 1)

            # ONLINE
            discrete_logits_online_split = self.modelAssembler.world_model.action_embedder.unembed(
                policy_embed_online,
                pred_head_index=0,
                return_split_discrete=True
            )  # list of 17 tensors, each (B, T, 2)

            discrete_logits_online = torch.stack(discrete_logits_online_split, dim=-2)  # (B, T, 17, 2)

            decoded_discrete_logits_online = ut.safe_cat(
                (decoded_discrete_logits_online, discrete_logits_online),
                dim=1
            )

            # LAGGING
            discrete_logits_lagging_split = self.modelAssembler.world_model.action_embedder.unembed(
                policy_embed_lagging,
                pred_head_index=0,
                return_split_discrete=True
            )

            discrete_logits_lagging = torch.stack(discrete_logits_lagging_split, dim=-2)  # (B, T, 17, 2)

            decoded_discrete_logits_lagging = ut.safe_cat((decoded_discrete_logits_lagging, discrete_logits_lagging), dim = 1)
        #re-predict actions
        self.modelAssembler.world_model.policy_head_online.train()
        self.modelAssembler.world_model.value_head.train()

        continues = torch.ones_like(decoded_rewards)
        lambda_returns = self.compute_lambda_returns_batched(decoded_rewards, decoded_values, continues)
        advantages = self.compute_advantages(lambda_returns, decoded_values)

        # Or use GAE for better results

        from torch.distributions import Categorical

        # logits: (B,T,17,2)
        pi_new = Categorical(logits=decoded_discrete_logits_online)
        logp_new = pi_new.log_prob(actions.long()).sum(dim=-1)          # (B,T)

        pi_ref = Categorical(logits=decoded_discrete_logits_lagging)

        # losses stay the same
        pos_mask = (advantages > 0).float()
        neg_mask = (advantages < 0).float()

        pos_loss = -(logp_new * pos_mask).sum() / (pos_mask.sum() + 1e-8)
        neg_loss = +(logp_new * neg_mask).sum() / (neg_mask.sum() + 1e-8)

        # KL: sum over keys, then average over batch/time
        kl_ref_new = torch.distributions.kl_divergence(pi_ref, pi_new).sum(dim=-1).mean()  # (B,T)->scalar






        value_logits_all = torch.cat(value_logits_all, dim=1)  # [B, T, num_bins]

        # Encode lambda_returns to twohot targets
        lambda_returns_encoded = self.modelAssembler.world_model.reward_encoder(
            lambda_returns
        )  # [B, T, num_bins] - target distribution

        # Compute negative log likelihood
        # Option 1: Cross entropy loss (standard for twohot)
        value_loss = F.cross_entropy(
            value_logits_all.reshape(-1, value_logits_all.shape[-1]),  # [B*T, num_bins]
            lambda_returns_encoded.reshape(-1, lambda_returns_encoded.shape[-1]),  # [B*T, num_bins]
            reduction='mean'
        )

        # Option 2: If your encoder outputs log probs directly
        # value_loss = -(lambda_returns_encoded * value_logits_all).sum(dim=-1).mean()

        # Total policy loss (PMPO - Equation 11)
        alpha = 0.5  # Balance positive and negative advantages
        beta = 0.3   # KL regularization weight (from paper)

        policy_loss = (
            (1 - alpha) * pos_loss +  # Negative advantage states
            alpha * neg_loss +         # Positive advantage states  
            beta * kl_ref_new         # KL with frozen prior
        )



        self.kl_div_loss_weight = 0.3
        total_loss = policy_loss + value_loss        

        # Zero gradients
        self.policy_optim.zero_grad()
        self.value_optim.zero_grad()

        # Backprop
        total_loss.backward()

        

        # Optional: gradient clipping (recommended for Dreamer)
        torch.nn.utils.clip_grad_norm_(
            self.modelAssembler.world_model.policy_head_online.parameters(), max_norm=100.0
        )
        torch.nn.utils.clip_grad_norm_(
            self.modelAssembler.world_model.value_head.parameters(), max_norm=100.0
        )

        # Update parameters
        self.policy_optim.step()
        self.value_optim.step()

        return {'total_loss': total_loss.item()}


    def align_policy_heads(self):
        self.world_model.Align_Policy_Heads()




    def trainerSchedule(self
    ):
        torch.cuda.reset_peak_memory_stats()
        self.modelAssembler.initModels()     
        
        self.initOptimizers()
        print(
            "Max reserved:",
            torch.cuda.max_memory_reserved() / 1024**2,
            "MB"
        )
        # myAgent.getTokModel()
        self.MakeDatasets()
        self.trainTokenizer(
            num_test_steps = numTestStep)



    def trainWM1(self):
        self.MakeDatasets()
        self.test()


    def createTrajectory(self,numTrajecories=2):


        self.modelAssembler.initModels()
        self.MakeDatasets()    
        experiences = []

        for _ in range(numTrajecories):
            xperience=self.generateTrajectory(
                time_steps=20,
                num_steps = 4,
                batch_size = 1,
                agent_index = 0,
                return_for_policy_optimization=True
            )

            xperience = xperience.cpu()
            experiences.append(xperience)

        losses = []

        for i, xperience in enumerate(experiences):
            xperience = xperience.cuda()
            loss = self.headTrainer(experience=xperience)
            losses.append(loss)
            print(f"Trained on experience {i+1}/{len(experiences)}, loss: {loss}")
        
        # loss=self.headTrainer(
        #     experience=xperience
        # )

        


if __name__ == "__main__":
    print("Starting trainer schedule...not")