import open3d as o3d
import os
import mitsuba as mi
import numpy as np
import math
import imageio
import shutil


mi.set_variant('scalar_rgb')

mesh_root = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0419_dual_cam_toy_example/output0/hands/visualization_meshes/"


hand_mesh_cam0 = os.path.join(mesh_root, "frame_1000_cam0_mesh.obj")
hand_mesh_cam1 = os.path.join(mesh_root, "frame_1000_cam1_mesh.obj")
object_mesh = os.path.join(mesh_root, "frame_1000_object_mesh.obj")
blended_mesh = os.path.join(mesh_root, "frame_1000_blended_mesh.obj")

# Define colors
color_cam0 = mi.Color3f(0.5, 0.7, 1.0)  # Light Blue
color_cam1 = mi.Color3f(0.8, 0.6, 0.9)  # Light Purple
color_object = mi.Color3f(0.5, 0.5, 0.5) # Grey
color_blended = mi.Color3f(0.6, 0.9, 0.6) # Light Green
color_floor = mi.Color3f(1.0, 1.0, 1.0) # White

# Create an alias for Transform4f
T = mi.Transform4f

# Create transforms
# Position light to better illuminate the scene
light_transform = T().translate([1, 1, 2]) @ T().scale([3, 3, 1])
floor_transform = T().translate([0, 0, 0]) @ T().scale([5, 5, 1])

# Create the scene dictionary
scene_dict = {
    'type': 'scene',

    # Integrator
    'integrator': {'type': 'path'},

    # Sensor (Camera)
    'sensor': {
        'type': 'perspective',
        'to_world': T().look_at(
            origin=[2, -2, 1.5],
            target=[0, 0, 0.3],
            up=[0, 0, 1]
        ),
        'fov': 15, # Field of view - decreased further from 30
        'sampler': {
            'type': 'independent',
            'sample_count': 500 # Increase for better quality/less noise
        },
        'film': {
            'type': 'hdrfilm',
            'width': 1024,
            'height': 768,
            'rfilter': {'type': 'gaussian'},
            'pixel_format': 'rgb'
        },
    },

    # Light Source (Area Light)
    'emitter1': {
        'type': 'rectangle',
        'to_world': light_transform,
         'emitter': {
             'type': 'area',
             'radiance': {'type': 'rgb', 'value': [1000.0, 1000.0, 1000.0]} # Increased light intensity
         }
    },

    # Environment Emitter (for white background)
    'environment': {
        'type': 'constant',
        'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
    },

    # Meshes
    'hand0': {
        'type': 'obj',
        'filename': hand_mesh_cam0,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': color_cam0}
        }
    },
    'hand1': {
        'type': 'obj',
        'filename': hand_mesh_cam1,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': color_cam1}
        }
    },
    'object': {
        'type': 'obj',
        'filename': object_mesh,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': color_object}
        }
    },
    'blended': {
        'type': 'obj',
        'filename': blended_mesh,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': color_blended}
        }
    },
}

# --- Multi-view Rendering Setup ---
n_frames = 60  # Number of frames for the video
radius = 3.5   # Radius of the circular camera path
height = 1.5   # Height of the camera
target = mi.Point3f(0, 0, 0.3) # Point the camera looks at
up = mi.Vector3f(0, 0, 1)       # Up vector for the camera
output_dir = "rendered_frames"
video_filename = "output_video.mp4"
fps = 15 # Frames per second for the video

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rendered_images = []

print("Starting multi-view rendering...")
for i in range(n_frames):
    angle = (2 * math.pi * i) / n_frames
    origin_x = radius * math.cos(angle)
    origin_y = radius * math.sin(angle)
    origin = mi.Point3f(origin_x, origin_y, height)

    # Update camera transform in the scene dictionary
    scene_dict['sensor']['to_world'] = T().look_at(origin=origin, target=target, up=up)

    # Load the scene with the updated camera pose
    scene = mi.load_dict(scene_dict)

    # Render the image
    print(f"Rendering frame {i+1}/{n_frames}...")
    image = mi.render(scene, spp=128) # spp overrides sampler sample_count

    # Save the frame
    frame_filename = os.path.join(output_dir, f"frame_{i:03d}.png")
    mi.util.write_bitmap(frame_filename, image)
    rendered_images.append(frame_filename)

print("Rendering complete.")

# --- Video Compilation ---
print(f"Compiling video {video_filename}...")
with imageio.get_writer(video_filename, fps=fps) as writer:
    for filename in rendered_images:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"Video saved to {video_filename}")

# --- Cleanup ---
print(f"Cleaning up temporary frame files in {output_dir}...")
shutil.rmtree(output_dir)

print("Process finished.")