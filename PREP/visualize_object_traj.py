import os
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import glob
import pickle
from scipy.spatial.transform import Rotation as R
import copy
import re
import argparse

def create_coordinate_frame(size=0.1):
    """Create a coordinate frame with xyz axes"""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return frame

def create_trajectory_line(points, color=[1, 0, 0]):
    """Create a line set from a sequence of points"""
    if len(points) < 2:
        return None
    
    # Create points for the line set
    line_points = o3d.utility.Vector3dVector(points)
    
    # Create lines connecting consecutive points
    lines = [[i, i+1] for i in range(len(points)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = line_points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def get_specific_marker_data(aruco_data, target_id):
    """
    Filter ArUco data to only include the marker with the specific target ID for each frame.
    
    Args:
        aruco_data: List of dictionaries, where each dict has 'frame', 'id', and 'transformation'
        target_id: The specific marker ID to look for
        
    Returns:
        List of dictionaries containing only the target ID marker data for each frame
    """
    filtered_data = []
    for frame_data in aruco_data:
        if not frame_data:  # Skip empty frames
            filtered_data.append(None)
            continue
            
        # Check if this frame has the target marker ID
        if frame_data["id"] == target_id:
            filtered_data.append(frame_data)
        else:
            filtered_data.append(None)
            
    return filtered_data

def visualize_trajectories(data_directory, save_directory, object_directory, trial_name, 
                         show_hand_frame=True, show_object_frame=True, fps=20):
    """
    Visualize trajectories of object marker, wrist marker, hand frame, and object frame
    along with hand point cloud.
    
    Args:
        data_directory: Path to data directory
        save_directory: Path to save directory
        object_directory: Name of object directory
        trial_name: Name of trial
        show_hand_frame: Whether to show hand frame and its trajectory
        show_object_frame: Whether to show object frame and its trajectory
        fps: Frames per second for output video
    """
    # Set up paths
    output_video_path = os.path.join(save_directory, trial_name, "output_trajectories")
    os.makedirs(output_video_path, exist_ok=True)
    
    # Load ArUco data
    aruco_data_path = os.path.join(data_directory, f"object_cam0/t{trial_name[-1]}_obj.pickle")
    wrist_aruco_data_path = os.path.join(data_directory, f"wrist_cam0/t{trial_name[-1]}_wri.pickle")
    
    print(f"Loading ArUco data from:")
    print(f"  Object: {aruco_data_path}")
    print(f"  Wrist: {wrist_aruco_data_path}")
    
    with open(aruco_data_path, "rb") as f:
        raw_aruco_data = pickle.load(f)
    with open(wrist_aruco_data_path, "rb") as f:
        raw_wrist_aruco_data = pickle.load(f)
    
    # Print data structure for debugging
    print("\nData structure debug:")
    if raw_aruco_data and len(raw_aruco_data) > 0:
        print("Object marker data structure example (first frame):")
        print(raw_aruco_data[0])
    if raw_wrist_aruco_data and len(raw_wrist_aruco_data) > 0:
        print("\nWrist marker data structure example (first frame):")
        print(raw_wrist_aruco_data[0])
    
    # Filter to only use specific marker IDs (0 for wrist, 1 for object)
    wrist_aruco_data = get_specific_marker_data(raw_wrist_aruco_data, 0)  # Wrist marker ID 0
    aruco_data = get_specific_marker_data(raw_aruco_data, 1)  # Object marker ID 1
    
    print(f"\nProcessing {len(aruco_data)} frames of ArUco data")
    print(f"Processing {len(wrist_aruco_data)} frames of wrist data")
    
    # Count valid frames (where both markers are visible)
    valid_frames = sum(1 for i in range(len(aruco_data)) 
                      if aruco_data[i] is not None and wrist_aruco_data[i] is not None)
    print(f"Found {valid_frames} frames with both markers visible")
    
    if valid_frames == 0:
        print("Error: No valid frames found with both markers visible!")
        return
    
    # Point cloud base path
    pc_base_path = os.path.join(save_directory, trial_name, "output0", "pointclouds")
    
    # Fine-tuning transformations
    CAM_INTRINSICS_EXTRINSICS_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/cam_intrinsics_extrinsics"
    fine_tune_T_path = os.path.join(CAM_INTRINSICS_EXTRINSICS_ROOT, f"{object_directory}_T.npy")
    if os.path.exists(fine_tune_T_path):
        fine_tune_T = np.load(fine_tune_T_path)
        print(f"Loaded fine-tuned object transformation from {fine_tune_T_path}")
    else:
        print(f"Warning: Fine-tuned object transformation not found at {fine_tune_T_path}, using default.")
        fine_tune_T = np.eye(4)
        fine_tune_T[:3, 3] = [0.025, 0.025, 0.01]
        fine_tune_T[:3, :3] = R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
    
    hand_fine_tune_T = np.eye(4)
    hand_fine_tune_T[:3, 3] = [0.1, 0.00, -0.01]
    hand_fine_tune_T[:3, :3] = R.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    
    # Setup camera
    ctr = vis.get_view_control()
    FRONT = [0, -1, 0]  # Look down from top
    UP = [0, 0, -1]     # Z up
    ZOOM = 0.5
    ctr.set_front(FRONT)
    ctr.set_up(UP)
    ctr.set_zoom(ZOOM)
    ctr.set_lookat([0, 0, 0])
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        os.path.join(output_video_path, "trajectories.mp4"),
        fourcc, fps, (1920, 1080)
    )
    
    # Store trajectories
    object_marker_traj = []
    wrist_marker_traj = []
    hand_frame_traj = [] if show_hand_frame else None
    object_frame_traj = [] if show_object_frame else None
    
    # Process each frame
    for frame_num in range(len(aruco_data)):
        # Check if markers are visible
        is_wrist_visible = (
            frame_num < len(wrist_aruco_data) and
            wrist_aruco_data[frame_num] is not None and
            wrist_aruco_data[frame_num]["transformation"] is not None
        )
        is_object_visible = (
            frame_num < len(aruco_data) and
            aruco_data[frame_num] is not None and
            aruco_data[frame_num]["transformation"] is not None
        )
        
        if not is_wrist_visible or not is_object_visible:
            continue
        
        # Get transformations from specific markers
        wrist_T = np.array(wrist_aruco_data[frame_num]["transformation"])
        object_T = np.array(aruco_data[frame_num]["transformation"])
        
        # Print marker IDs for first frame only
        if frame_num == 0:
            print(f"Using wrist marker ID: {wrist_aruco_data[frame_num]['id']} (should be 0)")
            print(f"Using object marker ID: {aruco_data[frame_num]['id']} (should be 1)")
        
        # Store trajectory points
        object_marker_traj.append(object_T[:3, 3])
        wrist_marker_traj.append(wrist_T[:3, 3])
        
        # Compute and store hand frame and object frame positions if enabled
        if show_hand_frame:
            hand_frame_T = np.dot(wrist_T, hand_fine_tune_T)
            hand_frame_traj.append(hand_frame_T[:3, 3])
        if show_object_frame:
            object_frame_T = np.dot(object_T, fine_tune_T)
            object_frame_traj.append(object_frame_T[:3, 3])
        
        # Load point cloud if available
        pc_frame_str = f"{frame_num:05d}"
        segmented_pc_path = os.path.join(pc_base_path, f"frame_{pc_frame_str}_masked.ply")
        if os.path.exists(segmented_pc_path):
            hand_pc = o3d.io.read_point_cloud(segmented_pc_path)
            hand_pc.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
            vis.add_geometry(hand_pc)
        
        # Create and add coordinate frames
        object_marker_frame = create_coordinate_frame(size=0.05)
        wrist_marker_frame = create_coordinate_frame(size=0.05)
        vis.add_geometry(object_marker_frame)
        vis.add_geometry(wrist_marker_frame)
        
        # Transform and add marker frames
        object_marker_frame.transform(object_T)
        wrist_marker_frame.transform(wrist_T)
        
        # Add hand frame and object frame if enabled
        if show_hand_frame:
            hand_frame = create_coordinate_frame(size=0.05)
            hand_frame.transform(hand_frame_T)
            vis.add_geometry(hand_frame)
            
        if show_object_frame:
            object_frame = create_coordinate_frame(size=0.05)
            object_frame.transform(object_frame_T)
            vis.add_geometry(object_frame)
        
        # Create and add trajectory lines
        if len(object_marker_traj) > 1:
            object_traj_line = create_trajectory_line(object_marker_traj, [1, 0, 0])  # Red
            wrist_traj_line = create_trajectory_line(wrist_marker_traj, [0, 1, 0])    # Green
            
            if object_traj_line: vis.add_geometry(object_traj_line)
            if wrist_traj_line: vis.add_geometry(wrist_traj_line)
            
            if show_hand_frame and len(hand_frame_traj) > 1:
                hand_traj_line = create_trajectory_line(hand_frame_traj, [0, 0, 1])  # Blue
                if hand_traj_line: vis.add_geometry(hand_traj_line)
                
            if show_object_frame and len(object_frame_traj) > 1:
                object_frame_traj_line = create_trajectory_line(object_frame_traj, [1, 1, 0])  # Yellow
                if object_frame_traj_line: vis.add_geometry(object_frame_traj_line)
        
        # Render frame
        vis.poll_events()
        vis.update_renderer()
        
        # Capture and save frame
        img = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(img)
        img = (img * 255).astype(np.uint8)
        video_writer.write(img)
        
        # Remove geometries for next frame
        vis.clear_geometries()
    
    # Cleanup
    video_writer.release()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trajectories of markers and frames')
    parser.add_argument('--object', type=str, default='spam_pla',
                      help='Object directory name (default: spam_real)')
    parser.add_argument('--trial', type=str, default='t1',
                      help='Trial name (default: t1)')
    parser.add_argument('--no-hand-frame', action='store_true',
                      help='Disable hand frame visualization')
    parser.add_argument('--no-object-frame', action='store_true',
                      help='Disable object frame visualization')
    parser.add_argument('--fps', type=int, default=20,
                      help='Frames per second for output video (default: 20)')
    
    args = parser.parse_args()
    
    DATA_ROOT = "/media/frida/Extreme SSD/sounding_hand/yuemin"
    SAVE_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/yuemin"
    
    data_directory = os.path.join(DATA_ROOT, args.object)
    save_directory = os.path.join(SAVE_ROOT, args.object)
    
    visualize_trajectories(
        data_directory, 
        save_directory, 
        args.object, 
        args.trial,
        show_hand_frame=not args.no_hand_frame,
        show_object_frame=not args.no_object_frame,
        fps=args.fps
    )
