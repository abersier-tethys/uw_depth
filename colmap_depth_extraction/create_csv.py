import os
import csv
from pathlib import Path

def create_dataset_csv(rgb_path, depth_path, output_csv='data/pg_dataset/dataset.csv'):
    """
    Creates a CSV file with three columns: image (rgb), depth, and features.csv
    """

    # Check if folders exist
    if not os.path.exists(rgb_path):
        print(f"Error: RGB folder not found")
        return
    
    if not os.path.exists(depth_path):
        print(f"Error: Depth folder not found")
        return
    
    # Get all files from both folders
    rgb_files = []
    depth_files = []

    # Read RGB files
    for file in os.listdir(rgb_path):
        if file.lower().endswith(('.png', '.jpg', '.tif', '.tiff')):
            rgb_files.append(file)
    
    # Read depth files
    for file in os.listdir(depth_path):
        if file.lower().endswith(('.tif', '.tiff')):
            depth_files.append(file)
    
    # Sort files to ensure consistent pairing
    rgb_files.sort()
    depth_files.sort()

    # Check if both folders have the same number of files
    if len(rgb_files) != len(depth_files):
        print(f"Error: RGB folder has {len(rgb_files)} files, depth folder has {len(depth_files)} files")
        return
    
    # Create CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write data rows
        for i in range(len(rgb_files)):
            rgb_path = f"data/pg_dataset/rgb/{rgb_files[i]}"
            depth_path = f"data/pg_dataset/depth/{depth_files[i]}"
            
            # Extract base filename (without extension) to create features filename
            base_name = Path(rgb_files[i]).stem
            features_path = f"data/pg_dataset/features/{base_name}_features.csv"
            
            writer.writerow([rgb_path, depth_path, features_path])
    
    print(f"CSV file '{output_csv}' created successfully with {len(rgb_files)} entries")


if __name__ == "__main__":
    # Create the CSV file
    create_dataset_csv()