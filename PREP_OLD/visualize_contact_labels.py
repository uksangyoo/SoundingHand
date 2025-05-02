import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import trimesh
import glob
import re
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.image as mpimg

def get_frame_number_from_filename(filename):
    """Extract frame number from filename."""
    match = re.search(r'frame_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    # Define paths - update these to match your dataset location
    DATA_NAME = "0419_dual_cam_toy_example"  # Update with your dataset name
    BASE_DIR = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand"  # Update with your base directory
    mesh_folder = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output0", "hands")
    output_video_path = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output0", "hand_contact_visualization.mp4")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Get all mesh files and corresponding contact labels
    mesh_files = sorted(glob.glob(os.path.join(mesh_folder, 'frame_*_hand_1.obj')), 
                      key=lambda x: get_frame_number_from_filename(x))
    
    # Define colormap for contact visualization
    colormap = plt.colormaps['coolwarm']
    
    # Initialize global min and max coordinates
    global_min_coords = None
    global_max_coords = None
    
    # First pass to determine global min and max coordinates
    for mesh_file in mesh_files:
        mesh = trimesh.load(mesh_file)
        vertices = np.array(mesh.vertices)
        if global_min_coords is None:
            global_min_coords = np.min(vertices, axis=0)
            global_max_coords = np.max(vertices, axis=0)
        else:
            global_min_coords = np.minimum(global_min_coords, np.min(vertices, axis=0))
            global_max_coords = np.maximum(global_max_coords, np.max(vertices, axis=0))
    
    # Add margins to the coordinates
    margin = 0.05
    global_min_coords -= margin
    global_max_coords += margin
    
    # Setup the figure and 3D axis for visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    ax_img = axes[0]
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # Remove the 2nd axis created by plt.subplots (axes[1])
    fig.delaxes(axes[1])
    
    # Frame text object for displaying frame number
    frame_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=14)

    # Function to plot a single frame
    def plot_frame(frame_idx):
        if frame_idx >= len(mesh_files):
            return
        
        # Clear previous frame
        ax.clear()
        ax_img.clear()
        
        # Get mesh file and corresponding frame number
        mesh_file = mesh_files[frame_idx]
        frame_number = get_frame_number_from_filename(mesh_file)
        
        # Load the hand mesh
        mesh = trimesh.load(mesh_file)
        vertices = np.array(mesh.vertices)
        
        # Load contact labels if they exist
        contact_label_file = os.path.join(mesh_folder, f"frame_{frame_number}_hand_contact_labels.npy")
        if os.path.exists(contact_label_file):
            contact_labels = np.load(contact_label_file)
        else:
            # If no labels exist, assume all vertices are non-contact
            contact_labels = np.zeros(len(vertices), dtype=np.uint8)
        
        # Separate contact and non-contact vertices
        contact_vertices = vertices[contact_labels == 1]
        non_contact_vertices = vertices[contact_labels == 0]
        
        # Plot non-contact vertices in blue
        if len(non_contact_vertices) > 0:
            ax.scatter(
                non_contact_vertices[:, 0],
                non_contact_vertices[:, 1],
                non_contact_vertices[:, 2],
                c='blue',
                s=10,
                alpha=0.5,
                label="Non-contact"
            )
        
        # Plot contact vertices in red
        if len(contact_vertices) > 0:
            ax.scatter(
                contact_vertices[:, 0],
                contact_vertices[:, 1],
                contact_vertices[:, 2],
                c='red',
                s=20,  # Larger points for contact vertices
                alpha=1.0,
                label="Contact"
            )
        
        # Update frame number text
        frame_text = ax.text2D(0.05, 0.95, f"Frame: {frame_number}", transform=ax.transAxes, fontsize=14)
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_title('Hand Mesh with Contact Points')
        
        # Add legend
        ax.legend()
        
        # Set consistent viewing angle and scale
        ax.view_init(elev=30, azim=45)
        
        # Determine axis limits based on all vertices
        ax.set_xlim(global_min_coords[0], global_max_coords[0])
        ax.set_ylim(global_min_coords[1], global_max_coords[1])
        ax.set_zlim(global_min_coords[2], global_max_coords[2])
        ax.invert_zaxis()  # Flip the Z axis
        
        # Load and show the corresponding image
        img_path = os.path.join(BASE_DIR, "DATA", DATA_NAME, "output", f"visualization_{frame_number:05d}.png")
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax_img.imshow(img)
            ax_img.axis('off')
            # ax_img.set_title('Original Image')
        else:
            ax_img.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=16)
            ax_img.axis('off')
        
        return frame_text,
    
    print(f"Found {len(mesh_files)} mesh files. Creating animation...")
    
    # Create animation
    frames = len(mesh_files)
    
    # Initialize writer for saving the animation
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=5000)
    
    with writer.saving(fig, output_video_path, dpi=100):
        # Generate and save each frame
        for i in tqdm(range(frames)):
            plot_frame(i)
            writer.grab_frame()
    
    print(f"Animation saved to {output_video_path}")

if __name__ == "__main__":
    main()