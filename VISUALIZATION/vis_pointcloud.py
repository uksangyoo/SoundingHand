import open3d as o3d
import numpy as np
import os

def vis_pointcloud(pointcloud_path):
    pointcloud = o3d.io.read_point_cloud(pointcloud_path)
    o3d.visualization.draw_geometries([pointcloud])

if __name__ == "__main__":
    pointcloud_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/campbell_pla/t1/output0/pointclouds/frame_01198_masked.ply"
    vis_pointcloud(pointcloud_path)
