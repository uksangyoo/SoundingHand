import open3d as o3d
import numpy as np
import copy
import sys
import os
from scipy.spatial.transform import Rotation as R
import pickle

# ---- CONFIG ----
# Base paths
DATA_ROOT = "/media/frida/Extreme SSD/sounding_hand/yuemin"
SAVE_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/yuemin"
OBJECT_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/Object"
EXTRINSICS_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/cam_intrinsics_extrinsics"

# List of objects to process
object_directories = ["campbell_pla", "campbell_real", "cheese", "cheezit", "clamp", 
                     "drill", "juice", "knife", "marker", "mug", "mustard", "ranch",
                     "scissors", "screwdriver", "spam_pla", "spam_real", "wrench"]

def process_object(object_name):
    print(f"\n=== Processing {object_name} ===")
    
    # Setup paths for this object
    dataset_path = os.path.join(SAVE_ROOT, object_name, "t1")
    extrinsics_path = os.path.join(EXTRINSICS_ROOT, "extrinsics.json")
    object_mesh_path = os.path.join(OBJECT_ROOT, f"{object_name}.stl")
    save_path = os.path.join(EXTRINSICS_ROOT, f"{object_name}_T.npy")
    aruco_data_path = os.path.join(DATA_ROOT, object_name, "object_cam0", "t1_obj.pickle")
    
    # Prompt for frame number
    frame_number = int(input(f"Enter frame number to visualize for {object_name}: ").strip())
    
    # Point cloud paths
    cam0_pcd_path = os.path.join(dataset_path, "output0", "pointclouds", f"frame_{frame_number:05d}_full.ply")
    cam1_pcd_path = os.path.join(dataset_path, "output1", "pointclouds", f"frame_{frame_number:05d}_full.ply")
    
    # ---- Load point clouds ----
    print(f"Loading point clouds for frame {frame_number}...")
    cam0_pcd = o3d.io.read_point_cloud(cam0_pcd_path)
    cam1_pcd = o3d.io.read_point_cloud(cam1_pcd_path)
    
    # ---- Load extrinsics ----
    import json
    with open(extrinsics_path, 'r') as f:
        extrinsics = json.load(f)
    T = np.array(extrinsics['T'])
    T_inv = np.linalg.inv(T)
    
    # ---- Transform cam1 to cam0 frame ----
    cam1_pcd.transform(T_inv)
    
    # ---- Load object mesh ----
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    object_mesh.compute_vertex_normals()
    
    # ---- Load ArUco marker transformation for this frame ----
    with open(aruco_data_path, "rb") as f:
        aruco_data = pickle.load(f)
    if frame_number >= len(aruco_data):
        print(f"Frame {frame_number} not in ArUco data (max {len(aruco_data)-1})")
        return False
    if aruco_data[frame_number]["transformation"] is None:
        print(f"No ArUco marker detected in frame {frame_number}")
        return False
    object_T = np.array(aruco_data[frame_number]["transformation"])
    
    print("Loaded ArUco marker transformation for frame", frame_number)
    print(object_T)
    # ---- Initial transformation (identity or load previous) ----
    fine_tune_T = np.eye(4)
    if os.path.exists(save_path):
        try:
            fine_tune_T = np.load(save_path)
            if fine_tune_T.shape != (4, 4):
                print(f"Warning: Loaded transformation from {save_path} is not 4x4, using identity.")
                fine_tune_T = np.eye(4)
            else:
                print(f"Loaded previous transformation from {save_path}")
        except Exception as e:
            print(f"Warning: Could not load transformation from {save_path} ({e}), using identity.")
            fine_tune_T = np.eye(4)
    
    # ---- Visualization and interaction ----
    print("\nControls:")
    print("  Arrow keys: translate X/Y")
    print("  0/1: translate Z up/down")
    print("  a/d: rotate Z | w/s: rotate X | q/e: rotate Y")
    print("  r: reset transformation")
    print("  q: save transformation and exit")
    print("  Enter: exit without saving\n")
    
    vis = FineTuneVisualizer(cam0_pcd, cam1_pcd, object_mesh, fine_tune_T, object_T)
    should_save, fine_tune_T = vis.run()
    print("Final transformation matrix:")
    print(fine_tune_T)
    if should_save:
        np.save(save_path, fine_tune_T)
        print(f"Saved transformation to {save_path}")
    else:
        print("Did not save transformation.")
    
    return True

# Open3D visualizer with key callbacks
class FineTuneVisualizer:
    def __init__(self, cam0_pcd, cam1_pcd, object_mesh, fine_tune_T, object_T):
        self.cam0_pcd = cam0_pcd
        self.cam1_pcd = cam1_pcd
        self.object_mesh = object_mesh
        self.fine_tune_T = fine_tune_T.copy()
        self.object_T = object_T.copy()
        self.mesh_handle = None
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.should_exit = False
        self.should_save = False

    def update_mesh(self):
        # Remove old mesh if present
        if self.mesh_handle is not None:
            self.vis.remove_geometry(self.mesh_handle, reset_bounding_box=False)
        # Transform and add new mesh
        mesh = copy.deepcopy(self.object_mesh)
        mesh.transform(self.fine_tune_T)
        mesh.transform(self.object_T)
        mesh.paint_uniform_color([0.8, 0.8, 0.2])
        self.mesh_handle = mesh
        self.vis.add_geometry(self.mesh_handle, reset_bounding_box=False)
        self.vis.update_geometry(self.mesh_handle)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        self.vis.create_window("Fine-tune Object Transform", width=1280, height=960)
        self.vis.add_geometry(self.cam0_pcd)
        self.vis.add_geometry(self.cam1_pcd)
        self.update_mesh()
        # Add coordinate frame
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.vis.add_geometry(axes)
        # Set background to white
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        # Register key callbacks
        self.register_callbacks()
        while not self.should_exit:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.vis.destroy_window()
        return self.should_save, self.fine_tune_T

    def register_callbacks(self):
        # Translation step (meters)
        t_step = 0.002
        # Rotation step (degrees)
        r_step = 1.0
        def translate(dx=0, dy=0, dz=0):
            self.fine_tune_T[:3, 3] += np.array([dx, dy, dz])
            print(f"Translate: {self.fine_tune_T[:3, 3]}")
            self.update_mesh()
        def rotate(axis, deg):
            rot = R.from_euler(axis, deg, degrees=True).as_matrix()
            T_rot = np.eye(4)
            T_rot[:3, :3] = rot
            # Apply rotation about object center
            center = np.mean(np.asarray(self.object_mesh.vertices), axis=0)
            T_center = np.eye(4)
            T_center[:3, 3] = -center
            T_uncenter = np.eye(4)
            T_uncenter[:3, 3] = center
            self.fine_tune_T = T_uncenter @ T_rot @ T_center @ self.fine_tune_T
            print(f"Rotate {axis} {deg} deg\n{self.fine_tune_T}")
            self.update_mesh()
        # Arrow keys: translate X/Y
        self.vis.register_key_callback(262, lambda vis: translate(dx= t_step)) # Right
        self.vis.register_key_callback(263, lambda vis: translate(dx=-t_step)) # Left
        self.vis.register_key_callback(264, lambda vis: translate(dy=-t_step)) # Down
        self.vis.register_key_callback(265, lambda vis: translate(dy= t_step)) # Up
        # 0/1: translate Z
        self.vis.register_key_callback(ord('0'), lambda vis: translate(dz= t_step)) # 0 key (up)
        self.vis.register_key_callback(ord('1'), lambda vis: translate(dz=-t_step)) # 1 key (down)
        # a/d: rotate Z
        self.vis.register_key_callback(ord('A'), lambda vis: rotate('z',  r_step))
        self.vis.register_key_callback(ord('D'), lambda vis: rotate('z', -r_step))
        # w/s: rotate X
        self.vis.register_key_callback(ord('W'), lambda vis: rotate('x',  r_step))
        self.vis.register_key_callback(ord('S'), lambda vis: rotate('x', -r_step))
        # q/e: rotate Y
        self.vis.register_key_callback(ord('Q'), lambda vis: rotate('y',  r_step))
        self.vis.register_key_callback(ord('E'), lambda vis: rotate('y', -r_step))
        # r: reset
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset())
        # Enter: save and exit
        self.vis.register_key_callback(257, lambda vis: self.save_and_exit())
        # ESC: exit without saving
        self.vis.register_key_callback(256, lambda vis: self.exit())

    def reset(self):
        print("Resetting transformation.")
        self.fine_tune_T = np.eye(4)
        self.update_mesh()
    def save_and_exit(self):
        print("Saving transformation and exiting...")
        self.should_save = True
        self.should_exit = True
    def exit(self):
        print("Exiting without saving...")
        self.should_exit = True

if __name__ == "__main__":
    print("Available objects:")
    for i, obj in enumerate(object_directories):
        print(f"{i+1}. {obj}")
    
    while True:
        try:
            choice = input("\nEnter object number to process (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                break
            
            obj_idx = int(choice) - 1
            if 0 <= obj_idx < len(object_directories):
                process_object(object_directories[obj_idx])
            else:
                print("Invalid object number. Please try again.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break 