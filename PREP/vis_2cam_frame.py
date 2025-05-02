import open3d as o3d
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import copy

dataset_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/campbell_pla/t1"
extrinsics_path = "camera_intrinsics_extrinsics/extrinsics.json"
frame_number = 100
use_saved = False

cam1_path = dataset_path + f"/output0/pointclouds/frame_{frame_number:05d}_masked.ply"
cam2_path = dataset_path + f"/output1/pointclouds/frame_{frame_number:05d}_masked.ply"


cam1_pcd = o3d.io.read_point_cloud(cam1_path)
cam2_pcd = o3d.io.read_point_cloud(cam2_path)

# Load extrinsics
with open(extrinsics_path, 'r') as f:
    extrinsics = json.load(f)

# Get transformation matrix from extrinsics
T = np.array(extrinsics['T'])
T_inv = np.linalg.inv(T)
#save T_inv to file
np.save("camera_intrinsics_extrinsics/cam1_to_cam0.npy", T_inv)
# Apply transformation to cam2 point cloud to align with cam1
cam2_transformed = cam2_pcd.transform(T_inv)

# Visualize both point clouds
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(cam1_pcd)
vis.add_geometry(cam2_transformed)

# Optional: Add coordinate frame for reference
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
vis.add_geometry(coordinate_frame)

# Set background to white for better visibility
opt = vis.get_render_option()
opt.background_color = np.asarray([1, 1, 1])

vis.run()
vis.destroy_window()


mesh_cam0 = o3d.io.read_triangle_mesh(dataset_path + f"/output0/hands/frame_{frame_number}_hand_1.obj")
mesh_cam1 = o3d.io.read_triangle_mesh(dataset_path + f"/output1/hands/frame_{frame_number}_hand_1.obj")

mesh_cam0.compute_vertex_normals()
mesh_cam1.compute_vertex_normals()
# Compute vertex-to-vertex error and find optimal transformation
vertices_cam0 = np.asarray(mesh_cam0.vertices)
vertices_cam1 = np.asarray(mesh_cam1.vertices)

# Compute centroids
centroid_cam0 = np.mean(vertices_cam0, axis=0)
centroid_cam1 = np.mean(vertices_cam1, axis=0)

# Center the points
centered_cam0 = vertices_cam0 - centroid_cam0
centered_cam1 = vertices_cam1 - centroid_cam1

# Compute covariance matrix
H = centered_cam1.T @ centered_cam0

# SVD decomposition
U, S, Vt = np.linalg.svd(H)

# Compute rotation matrix
R = Vt.T @ U.T

# Handle reflection case
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T

# Compute translation
t = centroid_cam0 - R @ centroid_cam1

# Create 4x4 transformation matrix
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t

print("Transformation matrix from cam1 to cam0:")
print(T)

# Apply transformation to verify alignment
mesh_cam1_transformed = copy.deepcopy(mesh_cam1)
if use_saved:
    T = np.load("camera_intrinsics_extrinsics/hand_cam1_to_cam0.npy")
    mesh_cam1_transformed.transform(T)
else:
    mesh_cam1_transformed.transform(T)
    np.save("camera_intrinsics_extrinsics/hand_cam1_to_cam0.npy", T)


# Find vertices where x < 0.1
vertices_cam0 = np.asarray(mesh_cam0.vertices)
red_indices = np.where(vertices_cam0[:, 0] < 0.1)[0]

# Create red color array for selected vertices
red_color = np.array([1, 0, 0])  # RGB for red
vertex_colors = np.tile(np.array([0.65, 0.74, 0.86]), (len(vertices_cam0), 1))  # Default light blue color
vertex_colors[red_indices] = red_color

# Assign colors to mesh
mesh_cam0.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# Sample 30 random indices excluding red_indices
all_indices = np.arange(len(vertices_cam0))
valid_indices = np.setdiff1d(all_indices, red_indices)
green_indices = np.random.choice(valid_indices, size=45, replace=False)

# Create green color array for selected vertices
green_color = np.array([0, 1, 0])  # RGB for green
vertex_colors[green_indices] = green_color

# Update mesh colors
mesh_cam0.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)


# Find vertices where z > -0.01
yellow_indices = np.where(vertices_cam0[:, 2] < -0.00)[0]

# Create yellow color array for selected vertices
yellow_color = np.array([1, 1, 0])  # RGB for yellow
vertex_colors[yellow_indices] = yellow_color

# Update mesh colors
mesh_cam0.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)


# Save indices and colors to npz file
np.savez(
    "camera_intrinsics_extrinsics/arap_indices.npz",
    static_indices=red_indices,  # red indices are static points
    control_indices=green_indices  # green indices are control points
)

# Compute mean squared error after alignment
aligned_vertices_cam1 = np.asarray(mesh_cam1_transformed.vertices)
mse = np.mean(np.sum((vertices_cam0 - aligned_vertices_cam1) ** 2, axis=1))
print(f"Mean squared error after alignment: {mse:.6f}")


axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
o3d.visualization.draw_geometries([mesh_cam0, mesh_cam1_transformed, axes])