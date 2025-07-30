
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from depth_estimation.utils.visualization import gray_to_heatmap
from torchvision.utils import save_image


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    depth_folder = os.path.join(script_dir, "depth")

    for index, depth_map in enumerate(os.listdir(depth_folder)):
        depth_path = os.path.join(depth_folder, depth_map)

        # Skip if it's a directory
        if os.path.isdir(depth_path):
            continue

        depth_image = np.array(Image.open(depth_path))

        # print(f"TYPE OF DEPTH MAP: {depth_image.dtype}")
        # print(f"SHAPE OF DEPTH MAP: {depth_image.shape}")

        depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)
        # print(f"SHAPE OF THE DEPTH TENSOR: {depth_tensor.shape}")

        heatmap = gray_to_heatmap(depth_tensor)
        heatmap_path = os.path.join(depth_folder, f"visualization/heatmap_{index+1}.png")
        save_image(heatmap, heatmap_path)

if __name__ == "__main__":
    main()