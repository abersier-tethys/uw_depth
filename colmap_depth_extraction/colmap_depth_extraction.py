import os
import sys
import csv
import subprocess
from pathlib import Path

from .create_csv import create_dataset_csv
from .render_depth_maps import render_pg_scene, visualize_rendered_depth_map


def depth_extraction():
    # Extract the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the root of the UW_depth repo, then down to data/pg_dataset/rgb
    dataset_path = os.path.join(script_dir, '..', 'data', 'pg_dataset')
    # Normalize the path to resolve the '..' components
    dataset_path = os.path.normpath(dataset_path)
    rgb_path = os.path.join(dataset_path, "rgb")
    depth_path = os.path.join(dataset_path,"depth")

    # The first step of this pipeline is the rendering of .tif depth maps form the 3d mesh
    mesh_path = os.path.join(dataset_path, "colmap_undistorted_output/texture_scene.ply")
    sparse_model_path = os.path.join(dataset_path, "colmap_undistorted_output/sparse")
    render_pg_scene(mesh_path=mesh_path, sparse_model_path=sparse_model_path, output_dir=depth_path)
    
    # Then, a csv file with the correct structure is created
    create_dataset_csv(rgb_path=rgb_path, depth_path=depth_path)

    # Finally one needs to obtain priors: run the helper script used to extract keypoints and match them
    extract_feature_path = os.path.join(script_dir, '..', 'helper_scripts', 'extract_dataset_features.py')
    extract_feature_path = os.path.normpath(extract_feature_path)

    # Execute the script
    try:
        result = subprocess.run([sys.executable, extract_feature_path], 
                            capture_output=True, 
                            text=True, 
                            check=True)
        
        print("Script output:", result.stdout)
        if result.stderr:
            print("Script errors:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code {e.returncode}")
        print("Error output:", e.stderr)

    # At this point, one is ready for inference!

if __name__ == "__main__":
    # Execute the pipeline
    depth_extraction()