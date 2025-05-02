import open3d as o3d
import numpy as np

# Load the mesh
mesh = o3d.io.read_triangle_mesh("/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/spam_pla.obj")
print(f"Original vertex count: {len(mesh.vertices)}")

# Subdivide the mesh to increase vertex count
num_subdivisions = 3 # Increase this for even more vertices
remeshed = mesh.subdivide_loop(num_subdivisions)
print(f"Remeshed vertex count: {len(remeshed.vertices)}")

# Optionally, save the remeshed mesh
output_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/spam_pla_remeshed.obj"
o3d.io.write_triangle_mesh(output_path, remeshed)
print(f"Remeshed mesh saved to {output_path}")
#visualize the remehsed mesh vertices as point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(remeshed.vertices)
o3d.visualization.draw_geometries([pcd])