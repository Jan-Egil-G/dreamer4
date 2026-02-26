#| default_exp _trainer

#| export
import torch
import matplotlib.pyplot as plt
# from dreamer4._core import build_tiny_pinp ad_dataset, make_tiny_dataloader
# from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel
# from dreamer4.trainers import VideoTokenizerTrainer, SimTrainer, cycle, BehaviorCloneTrainer
CPU = False
neps = 10
device = 'cuda' if not CPU else 'cpu'
# dataset, env = build_tiny_pin pad_dataset(device=device, episode_length=100, n=neps)
import random
from typing import Optional, Literal, Dict, Any, List
from envs.pinpad import PinPad, MotionPlannerPinPad

import torch
from torch.utils.data import Dataset, TensorDataset


# loadWeights=False
# saveWeights=False
# saveVideo=False
# tok_tr_steps=20
# bc_tr_steps=20
# runs=8


# # from chatgpt
# dim, dim_latent = 512, 256

# tk = VideoTokenizer(
#     dim=dim,
#     dim_latent=dim_latent,
#     patch_size=16,                 # was 16
#     num_latent_tokens=4,          # was 4
#     encoder_depth=6,              # was default 4
#     decoder_depth=6,              # was default 4
#     time_block_every=4,           # was 4
#     attn_dim_head=32,             # was 16
#     lpips_loss_weight=0.0005,        # was 0.0
# ).to(device)


# # pinpad DS params
# N_TOK_EPISODES = 12
# N_BC_EPISODES = 12
# EPISODE_LENGTH = 110
# DEVICE = 'cuda'
# USE_MOTION_PLANNER = True
# RAND_LIM = 0.5

# dynamics = DynamicsWorldModel(
#     video_tokenizer=tk,
#     dim=dim,
#     dim_latent=dim_latent,
#     max_steps=64,
#     num_tasks=1,
#     num_latent_tokens=4,          # match tokenizer
#     num_spatial_tokens=4,         # was 2 (too tiny)
#     depth=6,                      # was default 4
#     time_block_every=4,           # was 4
#     pred_orig_latent=True,
#     num_discrete_actions=env.action_space.n,
#     attn_dim_head=32,
#     prob_no_shortcut_train=0.4,   # give shortcut some presence
#     num_residual_streams=1
# ).to(device)



# 2) Train tokenizer (VideoTrokenizerTrainer expects a TensorDataset and uses [0])
batch_size = 9
class NormalizeObsWrapper:
    """
    Wraps a PinPad env that returns (C,H,W) uint8/float in 0..255 and
    converts observations to float32 in [0,1]. Everything else is passthrough.
    """
    def __init__(self, env):
        self.env = env
        # expose gym-ish interface
        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)
        self.device = getattr(env, "device", "cpu")
        self.size = getattr(env, "size", (64, 64))
        self.task = getattr(env, "task", "three")

    def _norm(self, obs):
        # obs is a torch.Tensor (C,H,W) from your PinPad
        obs = obs.to(torch.float32)
        # If it looks like 0..255, scale to 0..1. Otherwise assume already normalized.
        if obs.max() > 1.0 or obs.min() < 0.0:
            obs = obs / 255.0
        return obs

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return self._norm(obs)

    def step(self, action):
        obs, r, done, truncated, info = self.env.step(action)
        return self._norm(obs), r, done, truncated, info

    # Optional: pass through render() etc.
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

class PinPadBCEpisodes(Dataset):
    def __init__(self, env, n_episodes=12, episode_length=16, use_motion_planner=True, return_length=16,randLim=0.5):
        self.samples = []
        H, W = env.size[0], env.size[1]

        self.episode_length=episode_length; self.return_length=return_length

        if use_motion_planner:
            mp = MotionPlannerPinPad(env,randLim)

        for _ in range(n_episodes):
            frames = []
            rewards = []
            actions = []

            obs = env.reset()                  # (C, H, W) float in 0..255
            for t in range(episode_length):
                frames.append(obs)              # store frame BEFORE action (standard)

                if use_motion_planner:
                    act = mp.sample()
                else:
                    act = env.action_space.sample()

                obs, r, done, _, _ = env.step(act)

                rewards.append(float(r))        # scalar
                actions.append(int(act))        # scalar int

                if done:                        # keep fixed length anyway
                    # pad the remainder by repeating last frame / zeros reward / no-op
                    for _pad in range(t + 1, episode_length):
                        frames.append(obs)
                        rewards.append(0.0)
                        actions.append(0)
                    break

            # Stack & reshape
            # frames: list of (C,H,W) -> (T,C,H,W) -> (C,T,H,W), normalize to [0,1]
            frames = torch.stack(frames, dim=0).float()  # (T,C,H,W)
            frames = frames.permute(1,0,2,3).contiguous()  # (C,T,H,W)
            frames = frames.detach().cpu()  # <-- keep CPU


            rewards = torch.tensor(rewards, dtype=torch.float32)  # (T,)
            discrete = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)  # (T,1)

            # sanity
            assert frames.shape[1] == episode_length
            assert rewards.shape[0] == episode_length
            assert discrete.shape[:2] == (episode_length, 1)

            self.samples.append({
                "video": frames,                # (C,T,H,W) float[0,1]
                "rewards": rewards,             # (T,)
                "discrete_actions": discrete,   # (T,1)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # support both storage styles
        vid_key = "video_uint8" if "video_uint8" in s else "video"
        v = s[vid_key]                 # (C,T,H,W)
        r = s["rewards"]               # (T,)
        a = s["discrete_actions"]      # (T,1)

        C, T, H, W = v.shape
        K = self.return_length
        if K > T:
            raise ValueError(f"return_length ({K}) > episode_length ({T})")

        t0 = random.randint(0, T - K)  # inclusive range
        t1 = t0 + K

        # slice time dimension correctly
        v = v[:, t0:t1]                # (C,K,H,W)
        r = r[t0:t1]                   # (K,)
        a = a[t0:t1]                   # (K,1)

        # normalize if stored as uint8
        if v.dtype == torch.uint8:
            v = v.float().div_(255.0)

        # keep memory friendly
        v = v.contiguous()
        r = r.contiguous()
        a = a.contiguous()

        return {"video": v, "rewards": r, "discrete_actions": a}
print("Defined PinPadBCEpisodes")

class TokFromBCEpisodes(Dataset):
    """
    Sliding windows over bc_ds.samples[*]["video"] without extra storage.
    Assumes bc_ds.samples[i]["video"] is (C,T,H,W) on CPU.
    """
    def __init__(self, bc_ds, window=32, stride=1):
        self.bc = bc_ds
        self.window = window
        self.stride = stride
        self.T = bc_ds.episode_length
        self.starts_per_ep = max(0, (self.T - window)//stride + 1)
        print(" ")

    def __len__(self):
        return len(self.bc.samples) * self.starts_per_ep

    def __getitem__(self, i):
        ep_idx = i // self.starts_per_ep
        start  = (i % self.starts_per_ep) * self.stride
        v = self.bc.samples[ep_idx]["video"]          # (C,T,H,W) CPU
        return v[:, start:start+self.window]          # (C,K,H,W) view


def build_pinpad_datasets_for_trainers(
    n_tok_episodes=12,
    n_bc_episodes=12,
    episode_length=32,
    device="cuda",
    use_motion_planner=True,
    normalize_in_env=True,
    randLim=0.5
):
    print("Building PinPad datasets...")
    print("tokenizer data; ", end=' ')
    base_env = PinPad('three', length=episode_length, extra_obs=False,
                      size=[64, 64], random_starting_pos=True, device=device)
    env = NormalizeObsWrapper(base_env) if normalize_in_env else base_env

    vids = []
    mp_tok = MotionPlannerPinPad(env,randLim) if use_motion_planner else None
    for _ in range(n_tok_episodes):
        frames = []
        obs = env.reset()
        for t in range(episode_length):
            frames.append(obs)
            act = mp_tok.sample() if mp_tok else env.action_space.sample()
            obs, r, done, _, _ = env.step(act)
            if done:
                for _pad in range(t + 1, episode_length):
                    frames.append(obs)
                break

        frames = torch.stack(frames, dim=1).contiguous()  # (C,T,H,W)
        if not normalize_in_env:
            frames = frames.float() / 255.0
        vids.append(frames)

    print("WM data; ", end=' ')
    base_env_bc = PinPad('three', length=episode_length, extra_obs=False,
                         size=[64, 64], random_starting_pos=True, device=device)
    env_bc = NormalizeObsWrapper(base_env_bc) if normalize_in_env else base_env_bc
    bc_ds = PinPadBCEpisodes(env_bc, n_episodes=n_bc_episodes,
                             episode_length=episode_length,
                             use_motion_planner=use_motion_planner,
                             randLim=randLim)

    tok_ds = TokFromBCEpisodes(bc_ds, window=16, stride=10)
    print(f"Built PinPad tokenizer dataset with {len(tok_ds)} samples from {n_bc_episodes} BC episodes.")
    return tok_ds, bc_ds, env, vids






# Later in the script, use the parameters


import imageio.v2 as imageio
import torch

def save_video_gif(video, path="pinpad.gif", fps=1):
    # (C,T,H,W) -> (T,H,W,C)
    if video.shape[0] in (1, 3):
        video = video.permute(1, 2, 3, 0)
    video = (video.clamp(0, 1) * 255).byte().cpu().numpy()

    imageio.mimsave(path, [frame for frame in video], duration=1.0/fps)
    return path

# if saveVideo:
#     for i, vid in enumerate(vids):
#         gif_path = f"vids/pinpad_tokvid_{i}.gif"
#         save_video_gif(vid, path=gif_path, fps=2)
#         print(f"Saved tokenization dataset video {i} to {gif_path}")
#     exit()





import torch
import matplotlib.pyplot as plt

def get_video_from_sample(sample):
    """Return a (C,T,H,W) or (T,C,H,W) tensor from dict/tuple/tensor samples."""
    if isinstance(sample, dict):
        v = sample.get("video", next((x for x in sample.values() if torch.is_tensor(x)), None))
    elif isinstance(sample, (list, tuple)):
        # common case: TensorDataset(videos) -> (video,)
        v = sample[0]
    else:
        v = sample
    if not torch.is_tensor(v):
        raise TypeError(f"Couldn't find a tensor video in sample of type {type(sample)}")
    return v

def canonicalize_cthw(v):
    """Ensure (C,T,H,W); convert dtype/range for visibility."""
    if v.ndim != 4:
        raise ValueError(f"Expected 4D video, got shape {tuple(v.shape)}")
    # If first dim isn't channels, assume (T,C,H,W) and permute
    if v.shape[0] not in (1,3):
        v = v.permute(1,0,2,3).contiguous()  # (T,C,H,W) -> (C,T,H,W)
    # Make float in [0,1] for viewing
    if v.dtype.is_floating_point:
        # if it looks zero-centered, de-normalize (heuristic)
        if v.min() < 0:
            v = (v * 0.5) + 0.5
        v = v.clamp(0,1)
    else:
        v = v.float() / 255.0
    return v

def peek_video(sample, title="sample", t=0):
    v = get_video_from_sample(sample)
    print(f"{title} raw:", v.shape, v.dtype, "min/max:", float(v.min()), float(v.max()))
    v = canonicalize_cthw(v)
    C,T,H,W = v.shape
    img = v[:, min(t,T-1)].permute(1,2,0).cpu().numpy()
    print(f"{title} canonical:", v.shape, v.dtype, "min/max:", float(v.min()), float(v.max()))
    plt.figure(figsize=(3,3)); plt.imshow(img); plt.title(f"{title} t={min(t,T-1)}"); plt.axis("off"); 
    plt.savefig("plots/mse_plot2.png")
    return v






    ckpt = {
        "dynamics": unwrapped_dynamics.state_dict(),
        "tokenizer": unwrapped_tk.state_dict(),
        "bc_optim": bc_tr.optim.state_dict(),
        "tok_optim": tok_tr.optim.state_dict(),
    }
    torch.save(ckpt, "checkpoint.pt")




class WrappedEnv:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs = self.env.reset()
        obs = obs / 255.0
        return obs

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = obs / 255.0
        return obs, reward, done, trunc, info

import cv2
def build_tiny_pinpad_dataset(device='cuda', n=1024, episode_length=8):
    env = PinPad('three', length=episode_length, extra_obs=False, size=[64, 64], random_starting_pos=True, device=device)
    mp = MotionPlannerPinPad(env)
    env = WrappedEnv(env)
    obs = env.reset()
    # print the obs stats
    # make a tiny dataset
    eps_containing_success = 0; eps = 0
    ep_contained_success = False
    videos = []; curr_video = []; total_reward = 0; ep_reward = 0
    while True:
        curr_video.append(obs)
        action = mp.sample()
        obs, reward, done, terminated, info = env.step(action); ep_reward += reward

        ep_contained_success = info['success'] or ep_contained_success
        if done or terminated:
            if ep_contained_success or random.random() > 0.5: # don't take it sometimes, HACK to get a higher success rate in the base dataset
                eps_containing_success += 1 if ep_contained_success else 0
                eps += 1
                videos.append(torch.stack(curr_video))
                total_reward += ep_reward

            # unallocate all the memory in curr_video
            del curr_video

            curr_video = []; ep_reward = 0
            ep_contained_success = False
            obs = env.reset()

        if eps >= n:
            break


    videos = torch.stack(videos)  # (n, t, c, h, w)
    videos = videos.permute(0, 2, 1, 3, 4)  # (n, c, t, h, w)

    print(f"Built tiny PinPad dataset with {videos.shape[0]} videos of shape {videos.shape[1:]} - (c, t, h, w). Success rate {eps_containing_success / eps:1.1%}")
    return torch.utils.data.TensorDataset(videos), env
    # return videos







__all__ = ['to_device', 'make_tiny_dataloader', 'batchify_video', 'TinyOverfitConfig', 'train_tiny_overfit', 'plot_losses',
           'WrappedEnv', 'build_tiny_pinpad_dataset']

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch

def make_tiny_dataloader(
    dataset: Dataset,
    n_items: int = 8,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Take the first n_items from `dataset` and build a tiny DataLoader
    for overfitting/debugging.
    """
    indices = list(range(min(n_items, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def batchify_video(video: torch.Tensor) -> torch.Tensor:
    """
    Ensure (b, c, t, h, w). If dataset returns (c, t, h, w), add batch dim.
    """
    if video.ndim == 4:  # (c, t, h, w)
        video = video.unsqueeze(0)
    assert video.ndim == 5, f"Expected 5D video tensor, got shape {video.shape}"
    return video





def plot_losses(losses: Sequence[dict], keys=("total", "recon", "lpips")):
    steps = [d["step"] for d in losses]
    for k in keys:
        vals = [d[k] for d in losses]
        plt.plot(steps, vals, label=k)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("VideoTokenizer tiny-overfit losses")
    plt.show()



