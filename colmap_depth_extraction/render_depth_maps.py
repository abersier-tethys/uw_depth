import pyrender
import trimesh
import pycolmap
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix.
    
    Args:
        qvec: quaternion [w, x, y, z] or [x, y, z, w] depending on convention
    
    Returns:
        3x3 rotation matrix
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def load_colmap_cameras(sparse_folder):

    reconstruction = pycolmap.Reconstruction(sparse_folder)
    cameras = reconstruction.cameras
    images = reconstruction.images

    return cameras, images

def visualize_rendered_depth_map(depth_path, rendering_path):
    script_dir = Path(__file__).parent

    # Read the .tif image, where depth are stored as float values
    depth_map = tifffile.imread(depth_path)
    
    # Create masked array where zero values are masked
    masked_depth = np.ma.masked_where(depth_map == 0, depth_map)
    
    # Create colormap and set bad color
    cmap = plt.cm.inferno_r.copy()
    cmap.set_bad(color='black')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(masked_depth, cmap=cmap)
    plt.colorbar(label='Depth Value')
    plt.title('Depth Map')
    
    plt.savefig(rendering_path, dpi=300, bbox_inches='tight')
    plt.close()

def render_pg_scene(mesh_path, sparse_model_path, output_dir):
    
    # Make sure the output directory exists and is empty
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):

            # Skip .gitkeep files
            if filename == '.gitkeep':
                continue
            
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    os.makedirs(output_dir, exist_ok=True)

    # Make sure the output directory for the visualization exists and is empty
    rendering_dir = os.path.join(output_dir, "renderings")
    if os.path.exists(rendering_dir):
        os.rmdir(rendering_dir)
    os.makedirs(rendering_dir, exist_ok=True)

    # Load the 3D mesh reconstruction
    openMVS_mesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(openMVS_mesh)

    # Create a pyrender scene and add the mesh to it
    scene = pyrender.Scene()
    scene.add(mesh)

    cameras, images = load_colmap_cameras(sparse_model_path)
    images_list = list(images.items())
    print(f"Found {len(images_list)} images to process")

    # Iterate through camera images
    for i in range(len(images_list)):
        image_id, image = images_list[i]
        cam_data = cameras[image.camera_id]
        
        # Create renderer
        renderer = pyrender.OffscreenRenderer(cam_data.width, cam_data.height)

        # Extracts focal length and principal point information (intrinsics).
        # Important: This code snippet assumes that the camera
        # is a PINHOLE camera (which is the case after the undistortion)
        fx = np.float32(cam_data.params[0])
        fy = np.float32(cam_data.params[1])
        cx = np.float32(cam_data.params[2])
        cy = np.float32(cam_data.params[3])

        # Extracts the pose of the image.
        # Note that pyrender poses transform from camera
        # to world coordinates while colmap poses transform
        # from world to camera.
        cam_from_world = image.cam_from_world()

        # Use the matrix() method to obtain the matric directly
        world_to_cam_matrix = cam_from_world.matrix()  # 3x4 matrix
        
        # Convert to 4x4 and invert to get camera-to-world
        world_to_cam_4x4 = np.eye(4)
        world_to_cam_4x4[:3, :] = world_to_cam_matrix
        
        # Invert to get camera-to-world transformation
        T = np.linalg.inv(world_to_cam_4x4)
        
        # Handle coordinate system differences
        T[:, 1:3] *= -1
        
        print(f"Transformation matrix shape: {T.shape}")
        
        # Set up camera
        pyrender_camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, zfar=800.0)
        cam_node = scene.add(pyrender_camera, pose=T)
        
        # Render
        _, depth_map = renderer.render(scene)

        # Save results
        depth_filename = os.path.join(output_dir, f"depth{image_id:04d}.tif")

        # Save depth (convert to millimeters and keep float)
        tifffile.imwrite(str(depth_filename), depth_map.astype(np.float32))

        # Visualize the rendered depth map
        rendering_path = os.path.join(rendering_dir, f"heatmap_depth00{image_id}.png")
        visualize_rendered_depth_map(depth_filename, rendering_path)
    
        # cleanup
        scene.remove_node(cam_node)
        renderer.delete()

    print(f"Rendering complete! Output saved to: {output_dir}")

if __name__ == "__main__":
   # Extract the depth renderings for the whole dataset
   render_pg_scene()
