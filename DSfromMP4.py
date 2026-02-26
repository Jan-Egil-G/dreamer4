import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import random
import numpy as np
from pathlib import Path

import cv2
import cv2
import numpy as np
from typing import List, Optional

def extract_frames(video_path,
                   start_frame: int = 0, 
                   end_frame: Optional[int] = None,
                   step: int = 1,
                   resize_shape: Optional[tuple] = (64, 64)) -> np.ndarray:
    """
    Extracts frames, resizes them, and converts to RGB.
    Returns a numpy array of shape (T, H, W, C).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        
        # Jump to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for current in range(start_frame, end_frame, step):
            # If step > 1, we might need to skip frames manually if the 
            # backend doesn't support rapid cap.set calls efficiently
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Resize if requested
            if resize_shape:
                # cv2.resize expects (width, height)
                frame = cv2.resize(frame, (resize_shape[0], resize_shape[1]), interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
            
            # If step > 1, skip frames
            for _ in range(step - 1):
                cap.grab() # grab() is faster than read() as it doesn't decode
                
        return np.array(frames) # Returns (T, H, W, C)
        
    finally:
        cap.release()


class MP4BCEpisodes(Dataset):
    """
    BC dataset loaded from MP4 files with same interface as NPZBCEpisodes.
    Actions and rewards are None since we only have video.
    """
    def __init__(self,   return_length=16,  resize=None):
        """
        Args:
            mp4_paths: list of paths to .mp4 files, or single path, or directory
            episode_length: if provided, truncate/pad episodes to this length
            return_length: length of windows to return in __getitemv__
            max_episodes: maximum number of videos to load (None = all)
            resize: tuple (width, height) to resize frames, e.g. (320, 180). None = no resize
        """
        self.path="/workspace/GF-Minecraft/GF-Minecraft/data_269/data_269/video/"

        self.videosampled=get_random_video_sample(self.path, sample_size=32)
        # Limit number of episodes if max_episodes specified
        
        

        self.return_length = return_length

        self.resize = resize
        
    
    def __len__(self):
        return len(self.videosampled)
    
    def __getitem__(self, idx):
        for i in range(5):
            video_path = os.path.join(self.path, self.videosampled[random.randint(0, 31)])
            
            # Check total frames and pick valid range
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames> 600:
                break
            if i==4:
                print("error")
                exit()
        
        min_start = 300
        max_start = total_frames - self.return_length
        
        if max_start <= min_start:
            raise ValueError(f"Video too short: {total_frames} frames, need at least {min_start + self.return_length}")
        
        start = random.randint(min_start, max_start)
        
        v = extract_frames(
            video_path, 
            start_frame=start, 
            end_frame=start + self.return_length, 
            step=1,
            resize_shape=self.resize
        )
        
        v = v.transpose(3, 0, 1, 2)
        C, T, H, W = v.shape
            
        if self.return_length > T:
            raise ValueError(f"Found {T} frames, but need {self.return_length}")

        return {
            "video": v.astype(np.float32) / 255.0
        }
    
    def initNewEpoch(self):
        # If you want to re-sample a different segment of the video each epoch, you can implement that logic here.
        # For example, you could randomly select a new start frame for each video in the dataset.
        self.videosampled=get_random_video_sample(self.path, sample_size=32)


def build_datasets_from_mp4(

    return_length=16,

    resize=None,  # Add resize parameter
):
    """
    Build bc_ds and tok_ds from MP4 files, matching the NPZ interface.
    
    Args:
        mp4_paths: list of .mp4 file paths, single path, or directory path
        episode_length: if provided, truncate/pad episodes to this length
        return_length: length of windows returned by bc_ds
        tok_window: window size for tokenizer dataset
        tok_stride: stride for tokenizer sliding windows
        max_episodes: maximum number of videos to load (None = all)
        resize: tuple (width, height) to resize frames, e.g. (320, 180)
    
    Returns:
        tok_ds: TokFromBCEgpisodes dataset
        bc_ds: MP4BCEpisdodes dataset
    """
    print("Building datasets from MP4 files...")
    
    # Create BC dataset from MP4
    bc_ds = MP4BCEpisodes(

        return_length=return_length,

        resize=resize,  # Pass resize
    )
    
    print(f"Loaded {len(bc_ds)} episodes from MP4 files.")
    


    return bc_ds


# # Example usage:
# if __name__ == "__main__":
#     # Single file
#     # tok_ds, bc_ds =  ("video.mp4", max_episodes=10)
    
#     # Directory of MP4s
#     # tok_ds, bc_ds = build_datas ets_from_mp4("/path/to/videos/", max_episodes=100)
    
#     # List of files
#     mp4_list = ["video1.mp4", "video2.mp4", "video3.mp4"]
#     tok_ds, bc_ds = build_data sets_from_mp4(
#         mp4_list,
#         episode_length=1000,
#         return_length=16,
#         tok_window=16,
#         tok_stride=10,
#         max_episodes=50  # Only use first 50 videos
#     )
    
#     # Test
#     print(f"\nDataset sizes:")
#     print(f"  bc_ds: {len(bc_ds)} episodes")
#     print(f"  tok_ds: {len(tok_ds)} samples")
    
#     # Get a sample
#     sample = bc_ds[0]
#     print(f"\nSample shapes:")
#     print(f"  video: {sample['video'].shape}")  # (C, K, H, W)

#     import cv2
import os
from typing import List, Tuple, Optional
import numpy as np



    
def get_video_length(video_path) -> Tuple[float, int]:
    """
    Get the length of the MP4 file in seconds and total frame count.
    
    Returns:
        Tuple of (duration_in_seconds, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    
    try:
        # Get frames per second
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get total number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        return duration, frame_count
        
    finally:
        cap.release()

import os
import random
from pathlib import Path

# Target directory

def get_random_video_sample(path, sample_size=32):
    # Convert string path to Path object
    p = Path(path)
    
    # List all .mp4 files in the directory (non-recursive)
    video_files = [f.name for f in p.glob("*.mp4") if f.is_file()]
    
    # Check if we have enough files to sample
    if len(video_files) < sample_size:
        print(f"Warning: Only found {len(video_files)} files. Returning all of them.")
        return video_files
    
    # Randomly sample 32 files
    return random.sample(video_files, sample_size)





# # Execute and print
# sampled_list = get_random_video_sample(target_dir)

# print(f"--- Sample of 32 MP4 Files ---")
# for i, filename in enumerate(sampled_list, 1):
#     leng=get_video_length(os.path.join(target_dir, filename))  # Just to check if we can read the file
#     print(f"{i}. {filename} - Duration: Frames: {leng[1]}")