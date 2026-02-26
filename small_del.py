import cv2
import os

video_dir = "/workspace/GF-Minecraft/GF-Minecraft/data_269/data_269/video"

videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

for i, filename in enumerate(videos, 1):
    path = os.path.join(video_dir, filename)
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frames < 600:
        os.remove(path)
        print(f"[{i}/{len(videos)}] DELETED {filename} ({frames} frames)")
    else:
        print(f"[{i}/{len(videos)}] OK {filename} ({frames} frames)")