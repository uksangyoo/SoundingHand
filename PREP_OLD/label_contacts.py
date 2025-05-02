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
import re
from scipy.spatial import cKDTree

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
    
    # Apply fine-tuning transformation first if provided
    if hand_fine_tune_T is not None:
        o3d_mesh.transform(hand_fine_tune_T)
    
    # Then apply wrist marker transformation if provided
    if wrist_T is not None:
        o3d_mesh.transform(wrist_T)
    
    return o3d_mesh

def get_frame_number_from_filename(filename):
    """Extract frame number from filename."""
    match = re.search(r'frame_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def label_hand_contacts(hand_mesh, object_mesh, contact_radius=0.005):
    """
    Label hand mesh vertices as contact (1) or not (0) based on distance to object mesh.
    Args:
        hand_mesh: open3d.geometry.TriangleMesh (registered hand mesh)
        object_mesh: open3d.geometry.TriangleMesh (registered object mesh)
        contact_radius: float, distance threshold for contact (in meters)
    Returns:
        contact_labels: np.ndarray of shape (N_vertices,), 0 for no contact, 1 for contact
    """
    hand_vertices = np.asarray(hand_mesh.vertices)
    object_vertices = np.asarray(object_mesh.vertices)
    # Build KDTree for object mesh vertices
    object_kdtree = cKDTree(object_vertices)
    # Query distances to nearest object vertex for each hand vertex
    dists, _ = object_kdtree.query(hand_vertices, k=1)
    contact_labels = (dists <= contact_radius).astype(np.uint8)
    return contact_labels

def visualize_hands(mesh_folder, output_video_path, object_mesh_path, aruco_data_path, wrist_aruco_data_path, fine_tune_T, hand_fine_tune_T=None, fps=20, marker_id=1, wrist_marker_id=0, pc_base_path=None, contact_radius=0.01):
    """
    Visualize hand meshes, object, and contacts over a sequence of frames, saving a video and contact labels.
    
    Args:
        mesh_folder: Path to the folder containing hand mesh files
        output_video_path: Path to save the output video
        object_mesh_path: Path to the object mesh file
        aruco_data_path: Path to the ArUco data pickle file
        wrist_aruco_data_path: Path to the wrist ArUco tracking data pickle file
        fine_tune_T: Fine-tuning transformation matrix for the object
        hand_fine_tune_T: Fine-tuning transformation matrix from wrist marker frame to hand mesh frame
        fps: Frames per second for the output video
        marker_id: ID of the ArUco marker to use for object transformation
        wrist_marker_id: ID of the ArUco marker on the wrist
        pc_base_path: Base path for point cloud files
        contact_radius: Distance threshold (in meters) for labeling hand-object contact (default: 0.01)
    """
    # Get all mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_folder, 'frame_*_hand_1.obj')), 
                       key=lambda x: get_frame_number_from_filename(x))

    print(mesh_files)
    print(f"Found {len(mesh_files)} mesh files")
    
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
    
    # Load wrist ArUco data and print debug info
    print(f"\nLoading wrist ArUco data from: {wrist_aruco_data_path}")
    with open(wrist_aruco_data_path, "rb") as f:
        wrist_aruco_data = pickle.load(f)
    print(f"Total frames in wrist ArUco data: {len(wrist_aruco_data)}")
    
    # Extract frame numbers from mesh files for processing
    frame_numbers = [get_frame_number_from_filename(mesh_path) for mesh_path in mesh_files]
    
    # Process each frame
    for i, mesh_path in enumerate(tqdm(mesh_files)):
        # Get frame number from mesh path
        frame_number = frame_numbers[i]

        # Find the corresponding ArUco data entry indices for this frame
        wrist_aruco_idx = min(frame_number, len(wrist_aruco_data) - 1)
        aruco_idx = min(frame_number, len(aruco_data) - 1)

        # --- Check if both markers are visible ---
        is_wrist_visible = (
            wrist_aruco_idx < len(wrist_aruco_data) and
            wrist_aruco_data[wrist_aruco_idx]["transformation"] is not None and
            wrist_aruco_data[wrist_aruco_idx]["id"] == wrist_marker_id
        )
        is_object_visible = (
            aruco_idx < len(aruco_data) and
            aruco_data[aruco_idx]["transformation"] is not None and
            aruco_data[aruco_idx]["id"] == marker_id
        )

        if not is_wrist_visible or not is_object_visible:
            print(f"Skipping frame {frame_number}: Wrist visible={is_wrist_visible}, Object visible={is_object_visible}")
            continue # Skip this frame if either marker is not detected

        print(f"\nProcessing frame {frame_number}")
        
        # Format the frame number for point cloud path with leading zeros (5 digits)
        pc_frame_str = f"{frame_number:05d}"
        
        # Use provided point cloud base path
        pc_path = os.path.join(pc_base_path, f"frame_{pc_frame_str}_full.ply") if pc_base_path else None
        segmented_pc_path = os.path.join(pc_base_path, f"frame_{pc_frame_str}_masked.ply") if pc_base_path else None
        
        # Load point clouds
        point_cloud = None
        hand_point_cloud = None
        if pc_path and os.path.exists(pc_path):
            point_cloud = o3d.io.read_point_cloud(pc_path)
            hand_point_cloud = o3d.io.read_point_cloud(segmented_pc_path)
            # Estimate normals for the hand point cloud
            hand_point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            # Orient normals consistently
            hand_point_cloud.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            vis.add_geometry(point_cloud)
        else:
            print(f"Point cloud not found: {pc_path}")
        
        # --- Get transformations (already checked for visibility) ---
        wrist_T = np.array(wrist_aruco_data[wrist_aruco_idx]["transformation"])
        object_T = np.array(aruco_data[aruco_idx]["transformation"])

        # Add wrist marker coordinate frame
        wrist_frame = create_coordinate_frame(size=0.1)
        wrist_frame.transform(wrist_T)
        vis.add_geometry(wrist_frame)

        # Add hand frame if fine-tuning transform is provided
        if hand_fine_tune_T is not None:
            print("\n=== Hand Frame Debug ===")
            hand_frame = create_coordinate_frame(size=0.1)
            combined_T = np.dot(wrist_T, hand_fine_tune_T)
            print("Combined transformation matrix:")
            print(combined_T)
            hand_frame.transform(combined_T)
            vis.add_geometry(hand_frame)
        
        # Load and transform mesh using wrist marker transformation
        mesh = load_and_transform_mesh(mesh_path, wrist_T, hand_fine_tune_T)
        
        # If we have the segmented hand point cloud, perform ICP registration
        if hand_point_cloud is not None:
            print("\n=== ICP Registration ===")
            # Convert mesh to point cloud for ICP
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = mesh.vertices
            mesh_pcd.estimate_normals()
            
            # Perform ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                mesh_pcd, hand_point_cloud,
                max_correspondence_distance=0.02,  # 5cm max correspondence distance
                init=np.eye(4),  # Start from current position
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            print("ICP fitness score:", icp_result.fitness)
            print("ICP RMSE:", icp_result.inlier_rmse)
            print("ICP transformation:\n", icp_result.transformation)
            
            # Apply ICP transformation to the mesh
            mesh.transform(icp_result.transformation)
        
        vis.add_geometry(mesh)
        
        # --- Transform and add object mesh ---
        object_marker_frame = create_coordinate_frame(size=0.1)
        object_marker_frame.transform(object_T)
        vis.add_geometry(object_marker_frame)

        transformed_object = copy.deepcopy(object_mesh)
        transformed_object.transform(fine_tune_T)
        transformed_object.transform(object_T)
        vis.add_geometry(transformed_object)

        # Compute contact labels
        contact_labels = label_hand_contacts(mesh, transformed_object, contact_radius=contact_radius)

        # Save contact labels as .npy in the mesh folder, with frame number
        contact_label_path = os.path.join(mesh_folder, f"frame_{frame_number}_hand_contact_labels.npy")
        np.save(contact_label_path, contact_labels)
        print(f"Saved contact labels to {contact_label_path}")

        # Color hand mesh based on contact
        colors = np.zeros_like(np.asarray(mesh.vertices))
        colors[contact_labels == 1] = [0, 1, 0]  # Green for contact
        colors[contact_labels == 0] = [1, 0, 0]  # Red for no contact
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
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
        vis.remove_geometry(wrist_frame)
        if hand_fine_tune_T is not None:
            vis.remove_geometry(hand_frame)
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

def visualize_frame(mesh_folder, frame_number, object_mesh_path, aruco_data_path, wrist_aruco_data_path, fine_tune_T, hand_fine_tune_T=None, hand_id=0, marker_id=1, wrist_marker_id=0, pc_base_path=None):
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
        pc_base_path: Base path for point cloud files
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
    
    # Format the frame number for point cloud path with leading zeros (5 digits)
    pc_frame_str = f"{frame_number:05d}"
    
    # Use provided point cloud base path
    pc_path = os.path.join(pc_base_path, f"frame_{pc_frame_str}_full.ply") if pc_base_path else None
    segmented_pc_path = os.path.join(pc_base_path, f"frame_{pc_frame_str}_masked.ply") if pc_base_path else None
    
    # Load point clouds
    point_cloud = None
    hand_point_cloud = None
    if pc_path and os.path.exists(pc_path):
        point_cloud = o3d.io.read_point_cloud(pc_path)
        hand_point_cloud = o3d.io.read_point_cloud(segmented_pc_path)
        # Estimate normals for the hand point cloud
        hand_point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        # Orient normals consistently
        hand_point_cloud.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        vis.add_geometry(point_cloud)
    else:
        print(f"Point cloud not found: {pc_path}")
    
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
                combined_T = np.dot(wrist_T, hand_fine_tune_T)
                print("Combined transformation matrix:")
                print(combined_T)
                hand_frame.transform(combined_T)
        
        # Load and transform mesh using wrist marker transformation
        mesh = load_and_transform_mesh(mesh_path, wrist_T, hand_fine_tune_T)
        
        # If we have the segmented hand point cloud, perform ICP registration
        if hand_point_cloud is not None:
            print("\n=== ICP Registration ===")
            # Convert mesh to point cloud for ICP
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = mesh.vertices
            mesh_pcd.estimate_normals()
            
            # Perform ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                mesh_pcd, hand_point_cloud,
                max_correspondence_distance=0.02,  # 5cm max correspondence distance
                init=np.eye(4),  # Start from current position
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            
            print("ICP fitness score:", icp_result.fitness)
            print("ICP RMSE:", icp_result.inlier_rmse)
            print("ICP transformation:\n", icp_result.transformation)
            
            # Apply ICP transformation to the mesh
            mesh.transform(icp_result.transformation)
        
        vis.add_geometry(mesh)
    else:
        print(f"Hand mesh not found: {mesh_path}")
    
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
    DATA_NAME = "0419_dual_cam_toy_example"
    OBJECT_NAME = "spam_pla"

    BASE_DIR = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand"
    DATA_DIR = os.path.join(BASE_DIR, "DATA", DATA_NAME)
    object_mesh_path = os.path.join(BASE_DIR, "DATA", "objects", OBJECT_NAME + ".stl")
    mesh_folder = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output0", "hands")
    output_video_path = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output0", "hand_visualization.mp4")
    aruco_data_path = os.path.join(DATA_DIR, "object_tracking/t1_cam0_object_tracking.pickle")
    wrist_aruco_data_path = os.path.join(DATA_DIR, "wrist_tracking/t1_cam0_wrist_tracking.pickle")
    
    # Define point cloud base path
    pc_base_path = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output0", "pointclouds")

    # Define fine-tuning transformation for object (identity matrix as initial guess)
    fine_tune_T = np.eye(4)
    # Example adjustments:
    fine_tune_T[:3, 3] = [0.025, 0.018, 0.02]  # Translation
    fine_tune_T[:3, :3] = R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
    
    # Define transformation from wrist marker frame to hand mesh frame
    hand_fine_tune_T = np.eye(4)
    # Example adjustments - you'll need to tune these values:
    hand_fine_tune_T[:3, 3] = [0.1, 0.00, -0.01]  # Translation between wrist marker and hand mesh frame
    hand_fine_tune_T[:3, :3] = R.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()  # Rotation between frames
    
    # Uncomment one of these lines to run either visualization
    visualize_hands(mesh_folder, output_video_path, object_mesh_path, aruco_data_path, wrist_aruco_data_path,
                     fine_tune_T, hand_fine_tune_T, marker_id=1, wrist_marker_id=0, pc_base_path=pc_base_path , contact_radius=0.01)
    # visualize_frame(mesh_folder, frame_number=590, 
    #                object_mesh_path=object_mesh_path, 
    #                aruco_data_path=aruco_data_path, 
    #                wrist_aruco_data_path=wrist_aruco_data_path,
    #                fine_tune_T=fine_tune_T,
    #                hand_fine_tune_T=hand_fine_tune_T,
    #                hand_id=1, 
    #                marker_id=1,
    #                wrist_marker_id=0,
    #                pc_base_path=pc_base_path)