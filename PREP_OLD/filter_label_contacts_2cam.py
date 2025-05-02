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

def compute_chamfer_distance(points1, points2):
    """
    Computes the average squared Chamfer distance between two point sets.

    Args:
        points1: np.ndarray of shape (N, 3) or o3d.geometry.PointCloud/TriangleMesh
        points2: np.ndarray of shape (M, 3) or o3d.geometry.PointCloud/TriangleMesh

    Returns:
        float: The average squared Chamfer distance. Returns np.inf if either point set is empty.
    """
    if isinstance(points1, o3d.geometry.TriangleMesh):
        p1 = np.asarray(points1.vertices)
    elif isinstance(points1, o3d.geometry.PointCloud):
        p1 = np.asarray(points1.points)
    elif isinstance(points1, np.ndarray):
        p1 = points1
    else:
        raise TypeError("points1 must be numpy array, PointCloud, or TriangleMesh")

    if isinstance(points2, o3d.geometry.TriangleMesh):
        p2 = np.asarray(points2.vertices)
    elif isinstance(points2, o3d.geometry.PointCloud):
        p2 = np.asarray(points2.points)
    elif isinstance(points2, np.ndarray):
        p2 = points2
    else:
        raise TypeError("points2 must be numpy array, PointCloud, or TriangleMesh")

    if p1.shape[0] == 0 or p2.shape[0] == 0:
        # Return infinity if either point set is empty, as distance is undefined/infinite
        return np.inf 

    # Build KDTree for faster nearest neighbor search
    kdtree1 = cKDTree(p1)
    kdtree2 = cKDTree(p2)

    # Find nearest neighbor squared distances
    dist1_sq, _ = kdtree2.query(p1, k=1, p=2) # p=2 for Euclidean distance
    dist2_sq, _ = kdtree1.query(p2, k=1, p=2)

    # Calculate average squared Chamfer distance
    # Use squared distances directly as it's common and avoids sqrt
    chamfer_dist_sq = np.mean(dist1_sq**2) + np.mean(dist2_sq**2) 

    return chamfer_dist_sq

def visualize_hands(mesh_folder_cam0, mesh_folder_cam1, output_video_path, object_mesh_path, aruco_data_path, wrist_aruco_data_path, fine_tune_T, hand_fine_tune_T_cam0=None, hand_fine_tune_T_cam1=None, fps=20, marker_id=1, wrist_marker_id=0, pc_base_path_cam0=None, pc_base_path_cam1=None, cam1_to_cam0=None, contact_radius=0.01, control_indices=None, static_indices=None, hand_cam1_to_cam0=None):
    """
    Visualize blended hand meshes, object, and contacts over a sequence of frames, saving a video and contact labels.
    """
    # Get all mesh files
    mesh_files_cam0 = sorted(glob.glob(os.path.join(mesh_folder_cam0, 'frame_*_hand_1.obj')), 
                       key=lambda x: get_frame_number_from_filename(x))
    mesh_files_cam1 = sorted(glob.glob(os.path.join(mesh_folder_cam1, 'frame_*_hand_1.obj')), 
                       key=lambda x: get_frame_number_from_filename(x))

    print(f"Found {len(mesh_files_cam0)} mesh files in cam0")
    print(f"Found {len(mesh_files_cam1)} mesh files in cam1")
    
    if not mesh_files_cam0:
        raise ValueError(f"No mesh files found in {mesh_files_cam0}")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    
    # Add world coordinate frame (larger size)
    world_frame = create_coordinate_frame(size=0.1)  # World frame
    # vis.add_geometry(world_frame)
    
    # Store camera parameters that we'll reuse for each frame
    FRONT = [0, -1, 0]  # Look down from top (negative Y direction)
    UP = [0, 0, -1]      # Z up for top-down view
    ZOOM = 0.5
    
    # Initial camera setup
    ctr = vis.get_view_control()
    ctr.set_front(FRONT)
    ctr.set_up(UP)
    ctr.set_zoom(ZOOM)
    ctr.set_lookat([0, 0, 0])  # Look at the origin
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path + "/generated_video.mp4", fourcc, fps, (1920, 1080))
    
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
    frame_numbers_cam0 = [get_frame_number_from_filename(mesh_path) for mesh_path in mesh_files_cam0]
    frame_numbers_cam1 = [get_frame_number_from_filename(mesh_path) for mesh_path in mesh_files_cam1]
    
    # Process each frame
    for i, mesh_path in enumerate(tqdm(mesh_files_cam0)):
        # Get frame number from mesh path
        frame_number = frame_numbers_cam0[i]

        # List to keep track of geometries added in this iteration for proper cleanup
        geometries_to_remove = []

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
            # No geometries added yet in this iteration, just continue
            continue # Skip this frame if either marker is not detected

        print(f"\nProcessing frame {frame_number}")
        
        # Format the frame number for point cloud path with leading zeros (5 digits)
        pc_frame_str = f"{frame_number:05d}"
        
        # Load point clouds from both cameras
        hand_pc_cam0 = None
        hand_pc_cam1 = None
        combined_hand_pc = o3d.geometry.PointCloud()
        point_clouds = []
        
        # Camera 0 point clouds
        if pc_base_path_cam0:
            pc_path_cam0 = os.path.join(pc_base_path_cam0, f"frame_{pc_frame_str}_full.ply")
            segmented_pc_path_cam0 = os.path.join(pc_base_path_cam0, f"frame_{pc_frame_str}_masked.ply")
            
            if os.path.exists(pc_path_cam0):
                pc_cam0 = o3d.io.read_point_cloud(pc_path_cam0)
                hand_pc_cam0 = o3d.io.read_point_cloud(segmented_pc_path_cam0)
                
                # Estimate normals for camera 0's hand point cloud
                hand_pc_cam0.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                hand_pc_cam0.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                
                point_clouds.append(pc_cam0)
                combined_hand_pc += hand_pc_cam0
        
        # Camera 1 point clouds
        if pc_base_path_cam1 and cam1_to_cam0 is not None:
            pc_path_cam1 = os.path.join(pc_base_path_cam1, f"frame_{pc_frame_str}_full.ply")
            segmented_pc_path_cam1 = os.path.join(pc_base_path_cam1, f"frame_{pc_frame_str}_masked.ply")
            
            if os.path.exists(pc_path_cam1):
                pc_cam1 = o3d.io.read_point_cloud(pc_path_cam1)
                hand_pc_cam1 = o3d.io.read_point_cloud(segmented_pc_path_cam1)
                
                # Transform camera 1's point clouds to camera 0's coordinate frame
                pc_cam1.transform(cam1_to_cam0)
                hand_pc_cam1.transform(cam1_to_cam0)
                
                # Estimate normals for camera 1's hand point cloud
                hand_pc_cam1.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                hand_pc_cam1.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                
                point_clouds.append(pc_cam1)
                combined_hand_pc += hand_pc_cam1
        
        # Add point clouds to visualizer
        for pc in point_clouds:
            vis.add_geometry(pc)
            geometries_to_remove.append(pc) # Track added geometry
        
        # Process hand meshes from both cameras
        meshes = []
        for camera_id, (mesh_folder, hand_fine_tune_T) in enumerate([(mesh_folder_cam0, hand_fine_tune_T_cam0), 
                                                                    (mesh_folder_cam1, hand_fine_tune_T_cam1)]):
            mesh_path = os.path.join(mesh_folder, f'frame_{frame_number}_hand_1.obj')
            
            if not os.path.exists(mesh_path):
                print(f"Hand mesh not found for camera {camera_id}: {mesh_path}")
                continue
                
            # Get wrist transformation
            wrist_T = np.array(wrist_aruco_data[wrist_aruco_idx]["transformation"])
            print(f"\n=== Wrist Marker Debug (Camera {camera_id}) ===")
            print("Wrist transformation matrix:")
            print(wrist_T)
            
            # Create a larger coordinate frame for better visibility
            wrist_frame = create_coordinate_frame(size=0.1)  # Wrist marker frame
            wrist_frame.transform(wrist_T)
            #vis.add_geometry(wrist_frame)
            
            if hand_fine_tune_T is not None:
                print(f"\n=== Hand Frame Debug (Camera {camera_id}) ===")
                hand_frame = create_coordinate_frame(size=0.1)  # Hand frame
                # Apply both transformations
                combined_T = np.dot(wrist_T, hand_fine_tune_T)
                print("Combined transformation matrix:")
                print(combined_T)
                hand_frame.transform(combined_T)
                #vis.add_geometry(hand_frame)
            
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()
            
            # For camera 1, apply hand_cam1_to_cam0 transformation first if available
            if camera_id == 1 and hand_cam1_to_cam0 is not None:
                print("\n=== Applying hand_cam1_to_cam0 transformation to camera 1 mesh ===")
                print("hand_cam1_to_cam0 transformation matrix:")
                print(hand_cam1_to_cam0)
                mesh.transform(hand_cam1_to_cam0)
            
            # Then apply wrist marker and fine-tuning transformations
            if hand_fine_tune_T is not None:
                mesh.transform(hand_fine_tune_T)
            if wrist_T is not None:
                mesh.transform(wrist_T)
                
            meshes.append(mesh)
        
        if len(meshes) == 2:
            # Get object transformation and transform object mesh
            object_T = np.array(aruco_data[aruco_idx]["transformation"])
            print("\n=== Object Marker Debug ===")
            print("Object marker transformation matrix:")
            print(object_T)
            
            # Add coordinate frame at object marker pose
            object_marker_frame = create_coordinate_frame(size=0.1)  # Object marker frame
            object_marker_frame.transform(object_T)
            #vis.add_geometry(object_marker_frame)
            
            # Create a copy of the object mesh and apply transformations
            transformed_object = copy.deepcopy(object_mesh)
            transformed_object.transform(fine_tune_T)
            transformed_object.transform(object_T)
            vis.add_geometry(transformed_object)
            geometries_to_remove.append(transformed_object) # Track added geometry

            # Blend the meshes
            print("\n=== Blending Meshes ===")
            blended_mesh = blend_meshes(
                meshes[0], meshes[1],
                hand_pc_cam0, hand_pc_cam1,
                combined_hand_pc,
                control_indices,
                static_indices
            )

            # --- Chamfer Distance Check ---
            CHAMFER_THRESHOLD_SQ = 0.0003 # Threshold for average *squared* distance (e.g., 0.01m -> 0.0001m^2)
            chamfer_dist_sq = compute_chamfer_distance(blended_mesh, combined_hand_pc)
            print(f"Frame {frame_number}: Avg Squared Chamfer distance = {chamfer_dist_sq:.6f}")

            if chamfer_dist_sq > CHAMFER_THRESHOLD_SQ or not combined_hand_pc.has_points():
                skip_reason = "High Chamfer distance" if combined_hand_pc.has_points() else "Empty combined point cloud"
                print(f"Skipping frame {frame_number}: {skip_reason} ({chamfer_dist_sq:.6f} > {CHAMFER_THRESHOLD_SQ})")
                # Clean up geometries added *in this iteration* before skipping
                for geom in geometries_to_remove:
                    vis.remove_geometry(geom, reset_bounding_box=False)
                continue # Skip the rest of the processing for this frame
            # --- End Chamfer Distance Check ---

            # Compute contact labels for blended mesh
            contact_labels = label_hand_contacts(blended_mesh, transformed_object, contact_radius=contact_radius)

            # Save contact labels as .npy in both mesh folders
            contact_label_path = os.path.join(output_video_path, f"frame_{frame_number}_hand_contact_labels_blended.npy")
            np.save(contact_label_path, contact_labels)
            print(f"Saved blended contact labels to {contact_label_path}")

            # Color blended mesh based on contact
            colors = np.zeros_like(np.asarray(blended_mesh.vertices))
            colors[contact_labels == 1] = [0, 1, 0]  # Green for contact
            colors[contact_labels == 0] = [1, 0, 0]  # Red for no contact
            blended_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            #save blended mesh
            o3d.io.write_triangle_mesh(os.path.join(output_video_path, f"frame_{frame_number}_hand_blended.ply"), blended_mesh)
            # Add blended mesh to visualizer (only if Chamfer check passed)
            vis.add_geometry(blended_mesh)
            geometries_to_remove.append(blended_mesh) # Track added geometry
        
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
        
        # Remove geometries added in this frame iteration
        for geom in geometries_to_remove:
            vis.remove_geometry(geom, reset_bounding_box=False) # Use False for efficiency within loop
    
    # Cleanup
    vis.update_renderer() # Update final view state once after loop
    video_writer.release()
    vis.destroy_window()

def opencv_to_open3d_transform():
    """Convert from OpenCV to Open3D coordinate system convention"""
    # Both point cloud and ArUco markers use OpenCV convention:
    # X right, Y down, Z forward
    # No conversion needed, return identity matrix
    return np.eye(4)

def blend_meshes(mesh_cam0, mesh_cam1, hand_pc_cam0, hand_pc_cam1, combined_hand_pc, control_indices, static_indices):
    """
    Blend two meshes based on their proximity to the point cloud at control points.
    First performs ICP to align each mesh with its corresponding camera's point cloud, then blends based on proximity.
    
    Args:
        mesh_cam0: o3d.geometry.TriangleMesh from camera 0
        mesh_cam1: o3d.geometry.TriangleMesh from camera 1
        hand_pc_cam0: o3d.geometry.PointCloud of hand from camera 0
        hand_pc_cam1: o3d.geometry.PointCloud of hand from camera 1
        combined_hand_pc: o3d.geometry.PointCloud of the combined hand point cloud
        control_indices: np.array of indices for control vertices
        static_indices: np.array of indices for static vertices
    
    Returns:
        o3d.geometry.TriangleMesh: The blended mesh
    """
    # First perform ICP for both meshes with their corresponding point clouds
    aligned_meshes = []
    
    # Align mesh_cam0 with hand_pc_cam0
    mesh_pcd_0 = o3d.geometry.PointCloud()
    mesh_pcd_0.points = mesh_cam0.vertices
    mesh_pcd_0.estimate_normals()
    
    print("\n=== ICP Registration Camera 0 ===")
    icp_result_0 = o3d.pipelines.registration.registration_icp(
        mesh_pcd_0, combined_hand_pc,
        max_correspondence_distance=0.01,  # 5mm is more reasonable for hand-scale alignment
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    

    
    aligned_mesh_0 = copy.deepcopy(mesh_cam0)
    aligned_mesh_0.transform(icp_result_0.transformation)
    aligned_meshes.append(aligned_mesh_0)
    
    # Align mesh_cam1 with hand_pc_cam1
    mesh_pcd_1 = o3d.geometry.PointCloud()
    mesh_pcd_1.points = mesh_cam1.vertices
    mesh_pcd_1.estimate_normals()
    
    print("\n=== ICP Registration Camera 1 ===")
    icp_result_1 = o3d.pipelines.registration.registration_icp(
        mesh_pcd_1, combined_hand_pc,
        max_correspondence_distance=0.01,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    

    
    aligned_mesh_1 = copy.deepcopy(mesh_cam1)
    aligned_mesh_1.transform(icp_result_1.transformation)
    aligned_meshes.append(aligned_mesh_1)
    
    # Now proceed with blending using aligned meshes
    mesh_cam0, mesh_cam1 = aligned_meshes
    
    # Convert point cloud to numpy array
    pc_points = np.asarray(combined_hand_pc.points)
    pc_kdtree = cKDTree(pc_points)
    
    # Get vertices from both meshes
    verts_cam0 = np.asarray(mesh_cam0.vertices)
    verts_cam1 = np.asarray(mesh_cam1.vertices)
    
    # Get distances from control points to nearest points in point cloud
    dists_cam0, _ = pc_kdtree.query(verts_cam0[control_indices])
    dists_cam1, _ = pc_kdtree.query(verts_cam1[control_indices])
    
    # Count how many control points are closer in each mesh
    cam0_closer = np.sum(dists_cam0 < dists_cam1)
    cam1_closer = np.sum(dists_cam1 < dists_cam0)

    
    print(f"Control points closer: cam0={cam0_closer}, cam1={cam1_closer}")
    
    # If cam1 has no closer control points, there will be no deformation
    if cam1_closer == 0:
        print("WARNING: No control points from camera 1 are closer to the point cloud")
        print("This will result in no deformation, and the output will be identical to mesh_cam0")
        return mesh_cam0
    
    print("Blending meshes using ARAP deformation")
    
    # Otherwise, perform ARAP deformation
    # Start with mesh_cam0 and deform it towards mesh_cam1 where cam1 is closer
    blended_mesh = copy.deepcopy(mesh_cam0)
    
    # Create constraint vertices and positions
    constraint_vertices = []
    constraint_positions = []
    
    # Add static constraints
    print(f"Adding {len(static_indices)} static constraints")
    for idx in static_indices:
        constraint_vertices.append(int(idx))
        constraint_positions.append(verts_cam0[idx].tolist())
    
    # Add control point constraints where cam1 is closer
    control_points_used = 0


    for i, ctrl_idx in enumerate(control_indices):
        if dists_cam1[i] < dists_cam0[i]:
            constraint_vertices.append(int(ctrl_idx))
            constraint_positions.append(verts_cam1[ctrl_idx].tolist())
            control_points_used += 1

    
    
    # Print debug info about constraint distance
    if control_points_used > 0:
        constraint_diffs = []
        for i, ctrl_idx in enumerate(control_indices):
            if dists_cam1[i] < dists_cam0[i]:
                diff = np.linalg.norm(verts_cam1[ctrl_idx] - verts_cam0[ctrl_idx])
                constraint_diffs.append(diff)
                print(f"Control point {ctrl_idx}: displacement = {diff:.6f}")
        
        if constraint_diffs:
            print(f"Average constraint displacement: {np.mean(constraint_diffs):.6f}")
            print(f"Max constraint displacement: {np.max(constraint_diffs):.6f}")

    # Convert to Open3D types
    constraint_vertices = o3d.utility.IntVector(constraint_vertices)
    constraint_positions = o3d.utility.Vector3dVector(constraint_positions)
    
    # Perform ARAP deformation
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        blended_mesh = blended_mesh.deform_as_rigid_as_possible(
            constraint_vertex_indices=constraint_vertices,
            constraint_vertex_positions=constraint_positions,
            max_iter=1000,
            energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,
            smoothed_alpha=0.01
        )
    

    max_disp = np.max(np.linalg.norm(np.asarray(blended_mesh.vertices) - np.asarray(mesh_cam0.vertices), axis=1))
    print(f"Max Vertex Displacement: {max_disp:.6f}")
    print("Average dists_cam1:", np.mean(dists_cam1))
    print("Average dists_cam0:", np.mean(dists_cam0))

    
    return blended_mesh

def visualize_frame(mesh_folder_cam0, mesh_folder_cam1, frame_num, output_video_path, object_mesh_path, aruco_data_path, wrist_aruco_data_path, fine_tune_T, hand_fine_tune_T_cam0=None, hand_fine_tune_T_cam1=None, fps=20, marker_id=1, wrist_marker_id=0, pc_base_path_cam0=None, pc_base_path_cam1=None, cam1_to_cam0=None, contact_radius=0.01, control_indices=None, static_indices=None, hand_cam1_to_cam0=None):
    """
    Modified visualize_frame function that uses mesh blending
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    
    # Add world coordinate frame (larger size)
    world_frame = create_coordinate_frame(size=0.1)  # World frame
    #vis.add_geometry(world_frame)
    
    # Setup camera to match OpenCV convention
    ctr = vis.get_view_control()
    FRONT = [0, -1, 0]  # Look down from top (negative Y direction)
    UP = [0, 0, -1]     # Z up for top-down view
    ZOOM = 0.5
    ctr.set_front(FRONT)
    ctr.set_up(UP)
    ctr.set_zoom(ZOOM)
    ctr.set_lookat([0, 0, 0])  # Look at the origin
    
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
    print(f"Requested frame number: {frame_num}")
    
    # Check if both markers are visible
    is_wrist_visible = (
        frame_num < len(wrist_aruco_data) and
        wrist_aruco_data[frame_num]["transformation"] is not None and
        wrist_aruco_data[frame_num]["id"] == wrist_marker_id
    )
    is_object_visible = (
        frame_num < len(aruco_data) and
        aruco_data[frame_num]["transformation"] is not None and
        aruco_data[frame_num]["id"] == marker_id
    )

    if not is_wrist_visible or not is_object_visible:
        print(f"Skipping frame {frame_num}: Wrist visible={is_wrist_visible}, Object visible={is_object_visible}")
        vis.destroy_window()
        return
    
    # Format the frame number for point cloud path with leading zeros (5 digits)
    pc_frame_str = f"{frame_num:05d}"
    
    # Load point clouds from both cameras
    hand_pc_cam0 = None
    hand_pc_cam1 = None
    combined_hand_pc = o3d.geometry.PointCloud()
    point_clouds = []
    
    # Camera 0 point clouds
    if pc_base_path_cam0:
        pc_path_cam0 = os.path.join(pc_base_path_cam0, f"frame_{pc_frame_str}_full.ply")
        segmented_pc_path_cam0 = os.path.join(pc_base_path_cam0, f"frame_{pc_frame_str}_masked.ply")
        
        if os.path.exists(pc_path_cam0):
            pc_cam0 = o3d.io.read_point_cloud(pc_path_cam0)
            hand_pc_cam0 = o3d.io.read_point_cloud(segmented_pc_path_cam0)
            
            # Estimate normals for camera 0's hand point cloud
            hand_pc_cam0.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            hand_pc_cam0.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            
            point_clouds.append(pc_cam0)
            combined_hand_pc += hand_pc_cam0
    
    # Camera 1 point clouds
    if pc_base_path_cam1 and cam1_to_cam0 is not None:
        pc_path_cam1 = os.path.join(pc_base_path_cam1, f"frame_{pc_frame_str}_full.ply")
        segmented_pc_path_cam1 = os.path.join(pc_base_path_cam1, f"frame_{pc_frame_str}_masked.ply")
        
        if os.path.exists(pc_path_cam1):
            pc_cam1 = o3d.io.read_point_cloud(pc_path_cam1)
            hand_pc_cam1 = o3d.io.read_point_cloud(segmented_pc_path_cam1)
            
            # Transform camera 1's point clouds to camera 0's coordinate frame
            pc_cam1.transform(cam1_to_cam0)
            hand_pc_cam1.transform(cam1_to_cam0)
            
            # Estimate normals for camera 1's hand point cloud
            hand_pc_cam1.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            hand_pc_cam1.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            
            point_clouds.append(pc_cam1)
            combined_hand_pc += hand_pc_cam1
    
    # Add point clouds to visualizer
    # for pc in point_clouds:
    #     vis.add_geometry(pc)
    vis.add_geometry(combined_hand_pc)
    # Process hand meshes from both cameras
    meshes = []
    for camera_id, (mesh_folder, hand_fine_tune_T) in enumerate([(mesh_folder_cam0, hand_fine_tune_T_cam0), 
                                                                (mesh_folder_cam1, hand_fine_tune_T_cam1)]):
        mesh_path = os.path.join(mesh_folder, f'frame_{frame_num}_hand_1.obj')
        
        if not os.path.exists(mesh_path):
            print(f"Hand mesh not found for camera {camera_id}: {mesh_path}")
            continue
            
        # Get wrist transformation
        wrist_T = np.array(wrist_aruco_data[frame_num]["transformation"])
        print(f"\n=== Wrist Marker Debug (Camera {camera_id}) ===")
        print("Wrist transformation matrix:")
        print(wrist_T)
        
        # Create a larger coordinate frame for better visibility
        wrist_frame = create_coordinate_frame(size=0.1)  # Wrist marker frame
        wrist_frame.transform(wrist_T)
        #vis.add_geometry(wrist_frame)
        
        if hand_fine_tune_T is not None:
            print(f"\n=== Hand Frame Debug (Camera {camera_id}) ===")
            hand_frame = create_coordinate_frame(size=0.1)  # Hand frame
            # Apply both transformations
            combined_T = np.dot(wrist_T, hand_fine_tune_T)
            print("Combined transformation matrix:")
            print(combined_T)
            hand_frame.transform(combined_T)
            #vis.add_geometry(hand_frame)
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        
        # For camera 1, apply hand_cam1_to_cam0 transformation first if available
        if camera_id == 1 and hand_cam1_to_cam0 is not None:
            print("\n=== Applying hand_cam1_to_cam0 transformation to camera 1 mesh ===")
            print("hand_cam1_to_cam0 transformation matrix:")
            print(hand_cam1_to_cam0)
            mesh.transform(hand_cam1_to_cam0)
        
        # Then apply wrist marker and fine-tuning transformations
        if hand_fine_tune_T is not None:
            mesh.transform(hand_fine_tune_T)
        if wrist_T is not None:
            mesh.transform(wrist_T)
            
        meshes.append(mesh)
    
    if len(meshes) == 2:
        # Get object transformation and transform object mesh
        object_T = np.array(aruco_data[frame_num]["transformation"])
        print("\n=== Object Marker Debug ===")
        print("Object marker transformation matrix:")
        print(object_T)
        
        # Add coordinate frame at object marker pose
        object_marker_frame = create_coordinate_frame(size=0.1)  # Object marker frame
        object_marker_frame.transform(object_T)
        #vis.add_geometry(object_marker_frame)
        
        # Create a copy of the object mesh and apply transformations
        transformed_object = copy.deepcopy(object_mesh)
        transformed_object.transform(fine_tune_T)
        transformed_object.transform(object_T)
        vis.add_geometry(transformed_object)

        # Blend the meshes
        print("\n=== Blending Meshes ===")
        blended_mesh = blend_meshes(
            meshes[0], meshes[1],
            hand_pc_cam0, hand_pc_cam1,
            combined_hand_pc,
            control_indices,
            static_indices
        )
        vis.add_geometry(meshes[0])
        #paint meshes[1] light purple
        meshes[1].paint_uniform_color([0.8, 0.6, 0.8])  # Light purple color
        vis.add_geometry(meshes[1])
        
        # Create visualization meshes directory
        vis_meshes_dir = os.path.join(mesh_folder_cam0, "visualization_meshes")
        os.makedirs(vis_meshes_dir, exist_ok=True)
        
        # Save meshes
        o3d.io.write_triangle_mesh(os.path.join(vis_meshes_dir, f"frame_{frame_num}_cam0_mesh.obj"), meshes[0])
        o3d.io.write_triangle_mesh(os.path.join(vis_meshes_dir, f"frame_{frame_num}_cam1_mesh.obj"), meshes[1])
        o3d.io.write_triangle_mesh(os.path.join(vis_meshes_dir, f"frame_{frame_num}_object_mesh.obj"), transformed_object)
        o3d.io.write_triangle_mesh(os.path.join(vis_meshes_dir, f"frame_{frame_num}_blended_mesh.obj"), blended_mesh)
        
        print(f"Saved visualization meshes to {vis_meshes_dir}")


        # Compute contact labels for blended mesh
        contact_labels = label_hand_contacts(blended_mesh, transformed_object, contact_radius=contact_radius)

        # Save contact labels as .npy in both mesh folders
        contact_label_path = os.path.join(mesh_folder_cam0, f"frame_{frame_num}_hand_contact_labels_blended.npy")
        np.save(contact_label_path, contact_labels)
        print(f"Saved blended contact labels to {contact_label_path}")

        # Color blended mesh based on contact
        colors = np.zeros_like(np.asarray(blended_mesh.vertices))
        colors[contact_labels == 1] = [0, 1, 0]  # Green for contact
        colors[contact_labels == 0] = [1, 0, 0]  # Red for no contact
        blended_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Add blended mesh to visualizer
        vis.add_geometry(blended_mesh)
    
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
    mesh_folder_cam0 = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output0", "hands")
    mesh_folder_cam1 = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output1", "hands")
    output_video_path = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output_merged")
    aruco_data_path = os.path.join(DATA_DIR, "object_tracking/t1_cam0_object_tracking.pickle")
    wrist_aruco_data_path = os.path.join(DATA_DIR, "wrist_tracking/t1_cam0_wrist_tracking.pickle")
    
    # Define point cloud base paths for both cameras
    pc_base_path_cam0 = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output0", "pointclouds")
    pc_base_path_cam1 = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output1", "pointclouds")

    # Load camera transformation matrix
    cam1_to_cam0 = np.load(os.path.join(DATA_DIR, "camera_intrinsics_extrinsics/cam1_to_cam0.npy"))

    # Define fine-tuning transformation for object (identity matrix as initial guess)
    fine_tune_T = np.eye(4)
    # Example adjustments:
    fine_tune_T[:3, 3] = [0.025, 0.025, 0.01]  # Translation
    fine_tune_T[:3, :3] = R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()

    # Example adjustments - you'll need to tune these values:
    hand_fine_tune_T_cam0 = np.eye(4)
    hand_fine_tune_T_cam0[:3, 3] = [0.1, 0.00, -0.01]  # Translation between wrist marker and hand mesh frame
    hand_fine_tune_T_cam0[:3, :3] = R.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()  # Rotation between frames

    hand_fine_tune_T_cam1 = np.eye(4)
    hand_fine_tune_T_cam1[:3, 3] = [0.1, 0.00, -0.01]  # Translation between wrist marker and hand mesh frame
    hand_fine_tune_T_cam1[:3, :3] = R.from_euler('xyz', [150, 5, 180], degrees=True).as_matrix() # Rotation between frames
    
    hand_cam1_to_cam0 = np.load("/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0419_dual_cam_toy_example/camera_intrinsics_extrinsics/hand_cam1_to_cam0.npy")

    #Load arap indices
    arap_indices = np.load(os.path.join(DATA_DIR, "camera_intrinsics_extrinsics/arap_indices.npz"))
    static_indices = arap_indices["static_indices"]
    control_indices = arap_indices["control_indices"]

    visualize_hands(mesh_folder_cam0, mesh_folder_cam1, output_video_path, 
                   object_mesh_path, aruco_data_path, wrist_aruco_data_path,
                   fine_tune_T, hand_fine_tune_T_cam0, hand_fine_tune_T_cam1, fps=20, marker_id=1, 
                   wrist_marker_id=0, pc_base_path_cam0=pc_base_path_cam0,
                   pc_base_path_cam1=pc_base_path_cam1, cam1_to_cam0=cam1_to_cam0,
                   contact_radius=0.01, control_indices=control_indices, static_indices=static_indices,
                   hand_cam1_to_cam0=hand_cam1_to_cam0)
    
    visualize_frame(mesh_folder_cam0, mesh_folder_cam1, 1000, output_video_path, 
                object_mesh_path, aruco_data_path, wrist_aruco_data_path,
                fine_tune_T, hand_fine_tune_T_cam0, hand_fine_tune_T_cam1, fps=20, marker_id=1, 
                wrist_marker_id=0, pc_base_path_cam0=pc_base_path_cam0,
                pc_base_path_cam1=pc_base_path_cam1, cam1_to_cam0=cam1_to_cam0,
                contact_radius=0.01, control_indices=control_indices, static_indices=static_indices,
                hand_cam1_to_cam0=hand_cam1_to_cam0)