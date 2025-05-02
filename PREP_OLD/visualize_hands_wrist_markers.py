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

def create_sphere_at_point(center, radius=0.02, color=[1, 0, 0]):
    """Create a sphere mesh at the specified point"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere

def load_and_transform_mesh(mesh_path, wrist_T=None, hand_fine_tune_T=None):
    """Load mesh and transform it using wrist marker pose and optional transformations
    
    Args:
        mesh_path: Path to the mesh file
        wrist_T: Transformation matrix from wrist ArUco marker
        hand_fine_tune_T: Fine-tuning transformation matrix from wrist marker frame to hand mesh frame
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Convert to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    
    # Apply wrist marker transformation if provided
    if wrist_T is not None:
        o3d_mesh.transform(wrist_T)
    
    # Apply fine-tuning transformation if provided
    if hand_fine_tune_T is not None:
        o3d_mesh.transform(hand_fine_tune_T)
    
    return o3d_mesh

def visualize_hands(mesh_folder, output_video_path, object_mesh_path, aruco_data_path, wrist_aruco_data_path, fine_tune_T, hand_fine_tune_T=None, fps=20, marker_id=1, wrist_marker_id=0):
    # Get all mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_folder, 'frame_*_hand_1.obj')), 
                       key=lambda x: int(x.split('frame_')[1].split('_')[0]))
    print(mesh_files)
    if not mesh_files:
        raise ValueError(f"No mesh files found in {mesh_folder}")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    
    # Add world coordinate frame (larger size)
    world_frame = create_coordinate_frame(size=0.1)  # World frame
    vis.add_geometry(world_frame)
    
    # Store camera parameters that we'll reuse for each frame
    FRONT = [0, -1, 0]  # Look down from top (negative Y direction)
    UP = [0, 0, -1]      # Z up for top-down view
    ZOOM = 0.5
    
    # Initial camera setup
    ctr = vis.get_view_control()
    ctr.set_front(FRONT)
    ctr.set_up(UP)
    ctr.set_zoom(ZOOM)
    # Set the point to look at (center of the scene)
    ctr.set_lookat([0, 0, 0])  # Look at the origin
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))
    
    # Load object mesh
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    object_mesh.compute_vertex_normals()
    
    # Load ArUco data
    with open(aruco_data_path, "rb") as f:
        aruco_data = pickle.load(f)
    
    # Load wrist ArUco data
    with open(wrist_aruco_data_path, "rb") as f:
        wrist_aruco_data = pickle.load(f)
    
    # Process each frame
    for mesh_path in tqdm(mesh_files):
        # Get frame number from mesh path
        frame_number = int(mesh_path.split('frame_')[1].split('_')[0])
        
        # Load point cloud if available
        pc_path = os.path.join(mesh_folder, 'pointclouds', f'frame_{frame_number}.ply')
        point_cloud = None
        if os.path.exists(pc_path):
            point_cloud = o3d.io.read_point_cloud(pc_path)
            vis.add_geometry(point_cloud)
        
        # Get wrist transformation if available
        wrist_T = None
        if (frame_number < len(wrist_aruco_data) and 
            wrist_aruco_data[frame_number]["transformation"] is not None and 
            wrist_aruco_data[frame_number]["id"] == wrist_marker_id):
            
            wrist_T = np.array(wrist_aruco_data[frame_number]["transformation"])
            
            # Add wrist marker coordinate frame
            wrist_frame = create_coordinate_frame(size=0.1)
            wrist_frame.transform(wrist_T)
            vis.add_geometry(wrist_frame)
            
            # Add hand frame if fine-tuning transform is provided
            if hand_fine_tune_T is not None:
                hand_frame = create_coordinate_frame(size=0.1)
                combined_T = np.dot(wrist_T, hand_fine_tune_T)
                hand_frame.transform(combined_T)
                vis.add_geometry(hand_frame)
        
        # Load and transform mesh using wrist marker transformation
        mesh = load_and_transform_mesh(mesh_path, hand_fine_tune_T, wrist_T)
        vis.add_geometry(mesh)
        
        # Transform and add object mesh if ArUco data exists for this frame
        if (frame_number < len(aruco_data) and 
            aruco_data[frame_number]["transformation"] is not None and 
            aruco_data[frame_number]["id"] == marker_id):
            
            # Get transformation matrix from ArUco data
            T = np.array(aruco_data[frame_number]["transformation"])
            
            # Add coordinate frame at object marker pose
            object_marker_frame = create_coordinate_frame(size=0.1)
            object_marker_frame.transform(T)
            vis.add_geometry(object_marker_frame)
            
            # Create a copy of the object mesh and apply transformations
            transformed_object = copy.deepcopy(object_mesh)
            # First apply fine-tuning transformation
            transformed_object.transform(fine_tune_T)
            # Then apply marker transformation
            transformed_object.transform(T)
            
            # Add transformed object to visualizer
            vis.add_geometry(transformed_object)
        
        # Reset camera view for consistency
        ctr = vis.get_view_control()
        ctr.set_front(FRONT)
        ctr.set_up(UP)
        ctr.set_zoom(ZOOM)
        
        # Render frame
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        img = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(img)
        img = (img * 255).astype(np.uint8)
        
        # Write frame to video
        video_writer.write(img)
        
        # Remove geometries for next frame
        if point_cloud is not None:
            vis.remove_geometry(point_cloud)
        vis.remove_geometry(mesh)
        if wrist_T is not None:
            vis.remove_geometry(wrist_frame)
            if hand_fine_tune_T is not None:
                vis.remove_geometry(hand_frame)
        if (frame_number < len(aruco_data) and 
            aruco_data[frame_number]["transformation"] is not None and 
            aruco_data[frame_number]["id"] == marker_id):
            vis.remove_geometry(object_marker_frame)
            vis.remove_geometry(transformed_object)
    
    # Cleanup
    video_writer.release()
    vis.destroy_window()

def opencv_to_open3d_transform():
    """Convert from OpenCV to Open3D coordinate system convention"""
    # Both point cloud and ArUco markers use OpenCV convention:
    # X right, Y down, Z forward
    # No conversion needed, return identity matrix
    return np.eye(4)

def visualize_frame(mesh_folder, frame_number, object_mesh_path, aruco_data_path, wrist_aruco_data_path, fine_tune_T, hand_fine_tune_T=None, hand_id=0, marker_id=1, wrist_marker_id=0):
    """
    Visualize both point cloud and mesh for a specific frame number.
    
    Args:
        mesh_folder: Path to the folder containing mesh and point cloud files
        frame_number: Frame number to visualize
        object_mesh_path: Path to the object mesh file
        aruco_data_path: Path to the ArUco data pickle file
        wrist_aruco_data_path: Path to the wrist ArUco tracking data pickle file
        fine_tune_T: Fine-tuning transformation matrix for the object
        hand_fine_tune_T: Fine-tuning transformation matrix from wrist marker frame to hand mesh frame
        hand_id: ID of the hand to visualize (0 or 1)
        marker_id: ID of the ArUco marker to use for object transformation
        wrist_marker_id: ID of the ArUco marker on the wrist
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    
    # Add world coordinate frame (larger size)
    world_frame = create_coordinate_frame(size=0.1)  # World frame
    vis.add_geometry(world_frame)
    
    # Setup camera to match OpenCV convention
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])  # Look towards positive Z
    ctr.set_up([0, -1, 0])     # Y down (OpenCV convention)
    ctr.set_zoom(0.2)
    
    # Load object mesh
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    object_mesh.compute_vertex_normals()
    
    # Load ArUco data
    with open(aruco_data_path, "rb") as f:
        aruco_data = pickle.load(f)
    
    # Load wrist ArUco data and print debug info
    print(f"\nLoading wrist ArUco data from: {wrist_aruco_data_path}")
    with open(wrist_aruco_data_path, "rb") as f:
        wrist_aruco_data = pickle.load(f)
    print(f"Total frames in wrist ArUco data: {len(wrist_aruco_data)}")
    print(f"Requested frame number: {frame_number}")
    
    # Print frame data
    if frame_number < len(wrist_aruco_data):
        print("\nWrist ArUco frame data:")
        print(wrist_aruco_data[frame_number])
    
    # Load point cloud
    pc_path = os.path.join(mesh_folder, 'pointclouds', f'frame_{frame_number}.ply')
    if os.path.exists(pc_path):
        point_cloud = o3d.io.read_point_cloud(pc_path)
        vis.add_geometry(point_cloud)
    
    # Load and transform hand mesh
    mesh_path = os.path.join(mesh_folder, f'frame_{frame_number}_hand_{hand_id}.obj')
    
    if os.path.exists(mesh_path):
        # Get wrist transformation if available
        wrist_T = None
        if (frame_number < len(wrist_aruco_data) and 
            wrist_aruco_data[frame_number]["transformation"] is not None and 
            wrist_aruco_data[frame_number]["id"] == wrist_marker_id):
            
            print("\n=== Wrist Marker Debug ===")
            wrist_T = np.array(wrist_aruco_data[frame_number]["transformation"])
            print("Wrist transformation matrix:")
            print(wrist_T)
            
            # Create a larger coordinate frame for better visibility
            wrist_frame = create_coordinate_frame(size=0.1)  # Wrist marker frame
            wrist_frame.transform(wrist_T)
            vis.add_geometry(wrist_frame)
            
            if hand_fine_tune_T is not None:
                print("\n=== Hand Frame Debug ===")
                hand_frame = create_coordinate_frame(size=0.1)  # Hand frame
                # Apply both transformations
                combined_T = np.dot(wrist_T,hand_fine_tune_T)
                print("Combined transformation matrix:")
                print(combined_T)
                hand_frame.transform(combined_T)
                vis.add_geometry(hand_frame)
        
        # Load and transform mesh using wrist marker transformation
        mesh = load_and_transform_mesh(mesh_path, hand_fine_tune_T,wrist_T)
        vis.add_geometry(mesh)
    
    # Add object marker frame if ArUco data exists for this frame
    if (frame_number < len(aruco_data) and 
        aruco_data[frame_number]["transformation"] is not None and 
        aruco_data[frame_number]["id"] == marker_id):
        
        print("\n=== Object Marker Debug ===")
        T = np.array(aruco_data[frame_number]["transformation"])
        print("Object marker transformation matrix:")
        print(T)
        
        # Add coordinate frame at object marker pose
        object_marker_frame = create_coordinate_frame(size=0.1)  # Object marker frame
        object_marker_frame.transform(T)
        vis.add_geometry(object_marker_frame)
        
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
    mesh_folder = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/wrist_aruco_toy_example/output/hands"
    output_video_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/wrist_aruco_toy_example/output/hand_visualization.mp4"
    object_mesh_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/objects/spam_pla.obj"
    aruco_data_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/wrist_aruco_toy_example/wrist_aruco_toy_example_objet_tracking.pkl"
    wrist_aruco_data_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/wrist_aruco_toy_example/wrist_aruco_toy_example_wrist_tracking.pkl"

    # Define fine-tuning transformation for object (identity matrix as initial guess)
    fine_tune_T = np.eye(4)
    # Example adjustments:
    fine_tune_T[:3, 3] = [-0.02, 0.005, -0.04]  # Translation
    fine_tune_T[:3, :3] = R.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
    
    # Define transformation from wrist marker frame to hand mesh frame
    hand_fine_tune_T = np.eye(4)
    # Example adjustments - you'll need to tune these values:
    hand_fine_tune_T[:3, 3] = [0.1, 0.00, -0.01]  # Translation between wrist marker and hand mesh frame
    hand_fine_tune_T[:3, :3] = R.from_euler('xyz', [180, 5, 180], degrees=True).as_matrix()  # Rotation between frames
    
    # Uncomment one of these lines to run either visualization
    visualize_hands(mesh_folder, output_video_path, object_mesh_path, aruco_data_path, wrist_aruco_data_path,
                   fine_tune_T, hand_fine_tune_T, marker_id=1, wrist_marker_id=0)
    # visualize_frame(mesh_folder, frame_number=1200, 
    #                object_mesh_path=object_mesh_path, 
    #                aruco_data_path=aruco_data_path, 
    #                wrist_aruco_data_path=wrist_aruco_data_path,
    #                fine_tune_T=fine_tune_T,
    #                hand_fine_tune_T=hand_fine_tune_T,
    #                hand_id=1, 
    #                marker_id=1,
    #                wrist_marker_id=0) 