import json
import urllib.request
import os
import random

with open("index.json") as f:
    index = json.load(f)

base = index["basedir"]
output_dir = "/workspace/GF-Minecraft/GF-Minecraft/data_269/data_269/video"
os.makedirs(output_dir, exist_ok=True)

samples = random.sample(index["relpaths"], 40)

for i, relpath in enumerate(samples, 1):
    url = base + relpath
    filename = os.path.join(output_dir, os.path.basename(relpath))
    
    if os.path.exists(filename):
        print(f"[{i}/40] Skipping {os.path.basename(relpath)} (already exists)")
        continue
    
    print(f"[{i}/40] Downloading {os.path.basename(relpath)}...")
    urllib.request.urlretrieve(url, filename)

print("Done!")