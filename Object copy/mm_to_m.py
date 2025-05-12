#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import os

# Input and output file paths
input_file = 'campbell_pla.stl'
output_file = os.path.join(os.path.dirname(input_file), 'campbell_pla.stl')

# Load the STL file
print(f"Loading STL file: {input_file}")
mesh = o3d.io.read_triangle_mesh(input_file)
mesh.compute_vertex_normals()
# Convert from mm to meters (scale by 0.001)
mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * 0.001)

# Save the converted mesh
print(f"Saving converted file to: {output_file}")
o3d.io.write_triangle_mesh(output_file, mesh)

#visualizae mesh vertices as point clouds
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
o3d.visualization.draw_geometries([pcd])

print("Conversion complete!")
