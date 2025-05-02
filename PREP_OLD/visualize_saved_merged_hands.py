import os
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import glob
import re
from tqdm import tqdm

def get_frame_number_from_filename(filename):
    """Extract frame number from filename."""
    match = re.search(r'frame_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def visualize_saved_merged_hands(mesh_folder_cam0, output_video_path, fps=20):
    """
    Create a video visualization of the pre-computed merged hand meshes with contact labels.
    """
    # Get all mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_folder_cam0, 'frame_*_hand_1.obj')), 
                       key=lambda x: get_frame_number_from_filename(x))

    print(f"Found {len(mesh_files)} mesh files")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))
    
    # Process each frame
    for mesh_path in tqdm(mesh_files):
        frame_number = get_frame_number_from_filename(mesh_path)
        
        # Get corresponding contact labels file
        contact_label_path = os.path.join(mesh_folder_cam0, f"frame_{frame_number}_hand_contact_labels_blended.npy")
        
        if not os.path.exists(contact_label_path):
            print(f"Skipping frame {frame_number}: Contact labels not found")
            continue
        
        # Load mesh and contact labels
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        contact_labels = np.load(contact_label_path)
        
        # Color mesh based on contact labels
        colors = np.zeros_like(np.asarray(mesh.vertices))
        colors[contact_labels == 1] = [0, 1, 0]  # Green for contact
        colors[contact_labels == 0] = [1, 0, 0]  # Red for no contact
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Clear previous geometries and add new mesh
        vis.clear_geometries()
        vis.add_geometry(mesh)
        
        # Set camera view
        ctr = vis.get_view_control()
        ctr.set_front([0, -1, 0])  # Look from front
        ctr.set_up([0, 0, 1])      # Z up
        ctr.set_zoom(0.7)          # Adjust zoom level
        
        # Render and capture frame
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(img)
        img = (img * 255).astype(np.uint8)
        
        # Write frame to video
        video_writer.write(img)
    
    # Cleanup
    video_writer.release()
    vis.destroy_window()

if __name__ == "__main__":
    # Define paths
    DATA_NAME = "0419_dual_cam_toy_example"
    BASE_DIR = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand"
    DATA_DIR = os.path.join(BASE_DIR, "DATA", DATA_NAME)
    
    # Define input/output paths
    mesh_folder_cam0 = os.path.join(DATA_DIR, "output0", "hands")
    output_video_path = os.path.join(DATA_DIR, "output0", "saved_merged_hands.mp4")
    
    # Run visualization
    visualize_saved_merged_hands(
        mesh_folder_cam0=mesh_folder_cam0,
        output_video_path=output_video_path,
        fps=20
    ) 