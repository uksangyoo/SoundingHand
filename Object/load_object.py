import open3d as o3d
import numpy as np

def load_object(path):
    mesh = o3d.io.read_triangle_mesh(path)
    return mesh


if __name__ == "__main__":
    object= "scissors.stl"
    mesh = load_object(object)

    #convert to m from mm
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / 1000)

    #print bounding box
    print("bounding box: ", mesh.get_axis_aligned_bounding_box())
    #print center
    print("center: ", mesh.get_center())

    # #save mesh as a stl file
    # o3d.io.write_triangle_mesh(object.replace(".ply", ".stl"), mesh)
    