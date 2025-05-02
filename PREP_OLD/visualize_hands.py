import os
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import glob
import trimesh
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation as R
import copy

def create_coordinate_frame(size=0.1):
    """Create a coordinate frame with xyz axes"""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return frame

def load_and_transform_mesh(mesh_path, cam_t_path, fine_tune_T=None):
    """Load mesh and transform it using camera translation and optional fine-tuning
    
    Args:
        mesh_path: Path to the mesh file
        cam_t_path: Path to the camera translation file
        fine_tune_T: Optional fine-tuning transformation matrix
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Convert to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    
    # Load camera translation
    cam_t = np.load(cam_t_path)
    
    # Transform mesh to camera frame
    # The mesh is already in camera coordinates, so we just need to apply the translation
    o3d_mesh.translate(cam_t)
    
    # Apply fine-tuning transformation if provided
    if fine_tune_T is not None:
        o3d_mesh.transform(fine_tune_T)
    
    return o3d_mesh

def visualize_hands(mesh_folder, output_video_path, object_mesh_path, aruco_data_path, fine_tune_T, hand_fine_tune_T=None, fps=20, marker_id=1):
    # Get all mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_folder, 'frame_*_hand_1.obj')), 
                       key=lambda x: int(x.split('frame_')[1].split('_')[0]))
    print(mesh_files)
    if not mesh_files:
        raise ValueError(f"No mesh files found in {mesh_folder}")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    
    # Add coordinate frame
    coord_frame = create_coordinate_frame(size=0.1)
    vis.add_geometry(coord_frame)
    
    # Setup camera
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])  # Camera looks down negative z-axis
    ctr.set_up([0, 1, 0])     # Y-axis points down
    ctr.set_zoom(0.2)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))
    
    # Load object mesh
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    object_mesh.compute_vertex_normals()
    
    # Load ArUco data
    with open(aruco_data_path, "rb") as f:
        aruco_data = pickle.load(f)
    
    # Process each frame
    for mesh_path in tqdm(mesh_files):
        # Get frame number from mesh path
        frame_number = int(mesh_path.split('frame_')[1].split('_')[0])
        
        # Get corresponding camera translation file
        cam_t_path = mesh_path.replace('.obj', '_cam_t.npy')
        if not os.path.exists(cam_t_path):
            continue
            
        # Load and transform mesh
        mesh = load_and_transform_mesh(mesh_path, cam_t_path, hand_fine_tune_T)
        
        # Add hand mesh to visualizer
        vis.add_geometry(mesh)
        
        # Transform and add object mesh if ArUco data exists for this frame with the specified marker ID
        if (frame_number < len(aruco_data) and 
            aruco_data[frame_number]["transformation"] is not None and 
            aruco_data[frame_number]["id"] == marker_id):
            
            # Get transformation matrix from ArUco data
            T = np.array(aruco_data[frame_number]["transformation"])
            
            # Create a copy of the object mesh and apply transformations
            transformed_object = copy.deepcopy(object_mesh)
            # First apply fine-tuning transformation
            transformed_object.transform(fine_tune_T)
            # Then apply marker transformation
            transformed_object.transform(T)
            
            # Add transformed object to visualizer
            vis.add_geometry(transformed_object)
        
        # Render frame
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        img = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(img)
        img = (img * 255).astype(np.uint8)
        
        # Write frame to video
        video_writer.write(img)
        
        # Remove meshes for next frame
        vis.remove_geometry(mesh)
        if (frame_number < len(aruco_data) and 
            aruco_data[frame_number]["transformation"] is not None and 
            aruco_data[frame_number]["id"] == marker_id):
            vis.remove_geometry(transformed_object)
    
    # Cleanup
    video_writer.release()
    vis.destroy_window()

def visualize_frame(mesh_folder, frame_number, object_mesh_path, aruco_data_path, fine_tune_T, hand_fine_tune_T=None, hand_id=0, marker_id=1):
    """
    Visualize both point cloud and mesh for a specific frame number.
    
    Args:
        mesh_folder: Path to the folder containing mesh and point cloud files
        frame_number: Frame number to visualize
        object_mesh_path: Path to the object mesh file
        aruco_data_path: Path to the ArUco data pickle file
        fine_tune_T: Fine-tuning transformation matrix for the object
        hand_fine_tune_T: Fine-tuning transformation matrix for the hand
        hand_id: ID of the hand to visualize (0 or 1)
        marker_id: ID of the ArUco marker to use for object transformation
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    
    # Add coordinate frame
    coord_frame = create_coordinate_frame(size=0.1)
    vis.add_geometry(coord_frame)
    
    # Setup camera
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])  # Camera looks down negative z-axis
    ctr.set_up([0, 1, 0])     # Y-axis points down
    ctr.set_zoom(0.2)
    
    # Load object mesh
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    object_mesh.compute_vertex_normals()
    
    # Load ArUco data
    with open(aruco_data_path, "rb") as f:
        aruco_data = pickle.load(f)
    
    # Load and transform hand mesh
    mesh_path = os.path.join(mesh_folder, f'frame_{frame_number}_hand_{hand_id}.obj')
    cam_t_path = mesh_path.replace('.obj', '_cam_t.npy')
    
    if os.path.exists(mesh_path) and os.path.exists(cam_t_path):
        mesh = load_and_transform_mesh(mesh_path, cam_t_path, hand_fine_tune_T)
        vis.add_geometry(mesh)
    
    # Load point cloud
    pc_path = os.path.join(mesh_folder, 'pointclouds', f'frame_{frame_number}.ply')
    if os.path.exists(pc_path):
        point_cloud = o3d.io.read_point_cloud(pc_path)
        vis.add_geometry(point_cloud)
    
    # Transform and add object mesh if ArUco data exists for this frame with the specified marker ID
    if (frame_number < len(aruco_data) and 
        aruco_data[frame_number]["transformation"] is not None and 
        aruco_data[frame_number]["id"] == marker_id):
        
        # Get transformation matrix from ArUco data
        T = np.array(aruco_data[frame_number]["transformation"])
        
        # Create a copy of the object mesh and apply transformations
        transformed_object = copy.deepcopy(object_mesh)
        # First apply fine-tuning transformation
        transformed_object.transform(fine_tune_T)
        # Then apply marker transformation
        transformed_object.transform(T)
        
        # Add transformed object to visualizer
        vis.add_geometry(transformed_object)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Define paths
    mesh_folder = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/PREP/hand_meshes"
    output_video_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/PREP/hand_visualization.mp4"
    object_mesh_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/PREP/hand_meshes/spam_pla.obj"
    aruco_data_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/data/0410_spam_pla_aruco/t2_aruco.pkl"
    
    # Define fine-tuning transformation for object (identity matrix as initial guess)
    fine_tune_T = np.eye(4)
    # Example adjustments:
    fine_tune_T[:3, 3] = [-0.0, 0, -0.0]  # Translation
    fine_tune_T[:3, :3] = R.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
    
    # Define fine-tuning transformation for hand (identity matrix as initial guess)
    hand_fine_tune_T = np.eye(4)
    # Example adjustments:
    hand_fine_tune_T[:3, 3] = [0.0,0.01, -0.37]  # Translation
    hand_fine_tune_T[:3, :3] = R.from_euler('xyz', [0, -0, -0], degrees=True).as_matrix()
    
    # Uncomment one of these lines to run either visualization
    #visualize_hands(mesh_folder, output_video_path, object_mesh_path, aruco_data_path, fine_tune_T, hand_fine_tune_T, marker_id=1)
    visualize_frame(mesh_folder, frame_number=45, 
                   object_mesh_path=object_mesh_path, 
                   aruco_data_path=aruco_data_path, 
                   fine_tune_T=fine_tune_T,
                   hand_fine_tune_T=hand_fine_tune_T,
                   hand_id=1, 
                   marker_id=1) 