import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import imageio
import tempfile
from PIL import Image
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import glob
import argparse
import sys

# Import model definitions from the main module
# Assuming the model.py file is in the same directory
from model_pose_contact import AudioMeshMultiTaskModel, AudioMeshDataset, collate_fn

def load_model(model_path, model_class, device):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        model_class: Model class to instantiate
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Use the same parameters as in the training script
    model = model_class(n_channels=5, n_mels=64, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def find_rgb_image(base_dir, frame_num):
    """
    Find RGB image for a given frame number.
    
    Args:
        base_dir: Base directory to search in
        frame_num: Frame number to find
        
    Returns:
        Path to RGB image or None if not found
    """
    # Format frame number with leading zeros
    frame_str = f"{frame_num:05d}"
    
    # Look for visualization image
    image_path = os.path.join(base_dir, f"visualization_{frame_str}.png")
    if os.path.exists(image_path):
        return image_path
    
    # Alternative pattern if the first one doesn't exist
    image_path = os.path.join(base_dir, f"rgb/frame_{frame_str}.png")
    if os.path.exists(image_path):
        return image_path
    
    return None

def create_spectrogram_plot(audio_data, n_channels=5):
    """
    Create a plot of audio spectrograms.
    
    Args:
        audio_data: Audio spectrogram tensor [channels, mels, time]
        n_channels: Number of audio channels
        
    Returns:
        Matplotlib figure with spectrograms
    """
    fig, axes = plt.subplots(n_channels, 1, figsize=(6, 8))
    
    # If there's only one channel, axes won't be an array
    if n_channels == 1:
        axes = [axes]
    
    for i in range(n_channels):
        # Get spectrogram for channel i
        spec = audio_data[i].numpy()
        
        # Plot spectrogram
        im = axes[i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(f"Channel {i+1}")
        
        # Add labels only to bottom plot
        if i == n_channels - 1:
            axes[i].set_xlabel("Time")
        
        axes[i].set_ylabel("Mel Bins")
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    return fig

def visualize_results(model, test_loader, output_dir, base_data_dir, device='cuda', specific_frames=None):
    """
    Visualize model predictions and create GIFs.
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        output_dir: Directory to save visualization outputs
        base_data_dir: Base directory where RGB images are stored
        device: Device to run the model on
        specific_frames: List of specific frame numbers to visualize (if None, visualize all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define custom color maps
    contact_cmap = LinearSegmentedColormap.from_list('contact', [(0, 'blue'), (0.5, 'lightblue'), (1, 'red')])
    error_cmap = plt.cm.jet
    
    # Process each batch
    frame_data = {}
    
    with torch.no_grad():
        for batch in test_loader:
            # Skip if we already have enough frames
            if specific_frames is not None and all(frame in frame_data for frame in specific_frames):
                break
                
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            
            # Get predictions
            contact_logits, predicted_displacements = model(audio, mesh)
            contact_probs = torch.sigmoid(contact_logits)
            
            # Process each sample in the batch
            vertices_per_sample = mesh.x.size(0) // len(batch['frames'])
            for i, frame_num in enumerate(batch['frames']):
                # Skip if we're only interested in specific frames and this isn't one of them
                if specific_frames is not None and frame_num not in specific_frames:
                    continue
                
                # Skip if we already processed this frame
                if frame_num in frame_data:
                    continue
                
                # Extract data for this sample
                start_idx = i * vertices_per_sample
                end_idx = start_idx + vertices_per_sample
                
                # Get vertices and labels
                gt_vertices = mesh.x[start_idx:end_idx].cpu().numpy()
                ref_vertices = mesh.ref_vertices[start_idx:end_idx].cpu().numpy()
                gt_displacements = mesh.displacements[start_idx:end_idx].cpu().numpy()
                pred_displacements = predicted_displacements[start_idx:end_idx].cpu().numpy()
                pred_vertices = ref_vertices + pred_displacements
                
                # Get contact probabilities and labels
                gt_contacts = mesh.y[start_idx:end_idx].cpu().numpy()
                pred_contacts = contact_probs[start_idx:end_idx].cpu().numpy()
                
                # Get edge indices for this sample
                edge_index = mesh.edge_index.cpu().numpy()
                sample_edges = []
                for j in range(edge_index.shape[1]):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    if start_idx <= src < end_idx and start_idx <= dst < end_idx:
                        sample_edges.append((src - start_idx, dst - start_idx))
                
                # Find RGB image for this frame
                rgb_path = find_rgb_image(base_data_dir, frame_num)
                
                # Get audio data for this sample
                audio_data = audio[i].cpu()
                
                frame_data[frame_num] = {
                    'frame': frame_num,
                    'gt_vertices': gt_vertices,
                    'pred_vertices': pred_vertices,
                    'gt_contacts': gt_contacts,
                    'pred_contacts': pred_contacts,
                    'displacement_error': np.sqrt(np.sum((pred_vertices - gt_vertices)**2, axis=1)),
                    'edges': sample_edges,
                    'rgb_path': rgb_path,
                    'audio_data': audio_data
                }
    
    # Create GIFs and visualizations for each frame
    for frame_num, data in frame_data.items():
        print(f"Creating visualization for frame {frame_num}...")
        
        # Create a directory for this frame
        frame_dir = os.path.join(output_dir, f"frame_{frame_num:05d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        # 1. Create a comprehensive static visualization
        create_comprehensive_visualization(data, os.path.join(frame_dir, f"visualization_{frame_num:05d}.png"), contact_cmap, error_cmap)
        
        # 2. Create a spin GIF of the mesh with contact points
        create_spin_gif(data, os.path.join(frame_dir, f"spin_{frame_num:05d}.gif"), contact_cmap)
        
        # 3. Create a comparison GIF with ground truth and prediction
        create_comparison_gif(data, os.path.join(frame_dir, f"comparison_{frame_num:05d}.gif"), contact_cmap, error_cmap)
        
        print(f"Visualizations for frame {frame_num} saved to {frame_dir}")
    
    # If we have multiple frames, create a temporal sequence visualization
    if len(frame_data) > 1:
        print("Creating temporal sequence visualization...")
        create_temporal_sequence(frame_data, os.path.join(output_dir, "temporal_sequence.gif"), contact_cmap)
        print(f"Temporal sequence saved to {os.path.join(output_dir, 'temporal_sequence.gif')}")

def create_comprehensive_visualization(data, output_path, contact_cmap, error_cmap):
    """
    Create a comprehensive static visualization including mesh, audio, and RGB image.
    
    Args:
        data: Frame data dictionary
        output_path: Path to save the visualization
        contact_cmap: Colormap for contact visualization
        error_cmap: Colormap for error visualization
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Ground truth mesh with contact points
    ax1 = fig.add_subplot(231, projection='3d')
    plot_mesh_with_contacts(ax1, data['gt_vertices'], data['edges'], data['gt_contacts'],
                           title=f"Ground Truth (Frame {data['frame']})",
                           colormap=contact_cmap)
    
    # 2. Predicted mesh with contact probabilities
    ax2 = fig.add_subplot(232, projection='3d')
    plot_mesh_with_contacts(ax2, data['pred_vertices'], data['edges'], data['pred_contacts'],
                           title="Predicted Contacts",
                           colormap=contact_cmap)
    
    # 3. Error visualization
    ax3 = fig.add_subplot(233, projection='3d')
    max_error = data['displacement_error'].max()
    normalized_errors = data['displacement_error'] / max_error if max_error > 0 else data['displacement_error']
    plot_mesh_with_contacts(ax3, data['pred_vertices'], data['edges'], normalized_errors,
                           title=f"Error (Max: {max_error:.4f})",
                           colormap=error_cmap)
    
    # 4. RGB image if available
    ax4 = fig.add_subplot(234)
    if data['rgb_path'] is not None and os.path.exists(data['rgb_path']):
        img = plt.imread(data['rgb_path'])
        ax4.imshow(img)
        ax4.set_title(f"RGB Image (Frame {data['frame']})")
        ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, "RGB Image Not Available", ha='center', va='center')
        ax4.set_title("RGB Image")
        ax4.axis('off')
    
    # 5. Audio spectrogram
    ax5 = fig.add_subplot(235)
    if data['audio_data'] is not None:
        # Get first channel spectrogram for simplicity
        spec = data['audio_data'][0].numpy()
        im = ax5.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax5.set_title("Audio Spectrogram (Channel 1)")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Mel Bins")
        plt.colorbar(im, ax=ax5)
    else:
        ax5.text(0.5, 0.5, "Audio Data Not Available", ha='center', va='center')
        ax5.set_title("Audio Spectrogram")
        ax5.axis('off')
    
    # 6. Full Audio Spectrogram (all channels)
    ax6 = fig.add_subplot(236)
    if data['audio_data'] is not None:
        # Plot average across channels
        avg_spec = np.mean(data['audio_data'].numpy(), axis=0)
        im = ax6.imshow(avg_spec, aspect='auto', origin='lower', cmap='viridis')
        ax6.set_title("Audio Spectrogram (Average)")
        ax6.set_xlabel("Time")
        ax6.set_ylabel("Mel Bins")
        plt.colorbar(im, ax=ax6)
    else:
        ax6.text(0.5, 0.5, "Audio Data Not Available", ha='center', va='center')
        ax6.set_title("Audio Spectrogram")
        ax6.axis('off')
    
    # Add text with metrics
    contact_accuracy = np.mean((data['pred_contacts'] > 0.5) == data['gt_contacts'])
    contact_precision = np.sum(((data['pred_contacts'] > 0.5) == 1) & (data['gt_contacts'] == 1)) / (np.sum((data['pred_contacts'] > 0.5) == 1) + 1e-8)
    contact_recall = np.sum(((data['pred_contacts'] > 0.5) == 1) & (data['gt_contacts'] == 1)) / (np.sum(data['gt_contacts'] == 1) + 1e-8)
    
    metrics_text = (f"Contact Metrics - Accuracy: {contact_accuracy:.4f}\n"
                    f"Precision: {contact_precision:.4f}, Recall: {contact_recall:.4f}\n"
                    f"Mean Displacement Error: {np.mean(data['displacement_error']):.4f}")
    
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def create_spin_gif(data, output_path, colormap, n_frames=36):
    """
    Create a GIF showing the mesh rotating 360 degrees.
    
    Args:
        data: Frame data dictionary
        output_path: Path to save the GIF
        colormap: Colormap for visualization
        n_frames: Number of frames in the GIF
    """
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    # Create frames with different view angles
    for i in range(n_frames):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh with contacts
        plot_mesh_with_contacts(ax, data['pred_vertices'], data['edges'], data['pred_contacts'],
                              title=f"Predicted Contacts (Frame {data['frame']})",
                              colormap=colormap)
        
        # Set view angle
        angle = i * (360 / n_frames)
        ax.view_init(elev=30, azim=angle)
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100)
        frame_paths.append(frame_path)
        plt.close(fig)
    
    # Create GIF
    with imageio.get_writer(output_path, mode='I', duration=0.1) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)

def create_comparison_gif(data, output_path, contact_cmap, error_cmap, n_frames=12):
    """
    Create a GIF toggling between ground truth, prediction, and error.
    
    Args:
        data: Frame data dictionary
        output_path: Path to save the GIF
        contact_cmap: Colormap for contact visualization
        error_cmap: Colormap for error visualization
        n_frames: Number of animation frames (repeats the cycle)
    """
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    # Create the three main visualization types
    views = [
        ("Ground Truth", data['gt_vertices'], data['gt_contacts'], contact_cmap),
        ("Predicted", data['pred_vertices'], data['pred_contacts'], contact_cmap),
        ("Error", data['pred_vertices'], data['displacement_error'], error_cmap)
    ]
    
    # Create n_frames by cycling through the views
    for i in range(n_frames):
        view_idx = i % len(views)
        title, vertices, values, cmap = views[view_idx]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        plot_mesh_with_contacts(ax, vertices, data['edges'], values,
                               title=f"{title} (Frame {data['frame']})",
                               colormap=cmap)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        if view_idx == 2:  # Error view
            cbar.set_label('Displacement Error')
        else:
            cbar.set_label('Contact Probability' if view_idx == 1 else 'Contact Truth')
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100)
        frame_paths.append(frame_path)
        plt.close(fig)
    
    # Create GIF
    with imageio.get_writer(output_path, mode='I', duration=0.5) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)

def create_temporal_sequence(frame_data, output_path, colormap, max_frames=10):
    """
    Create a GIF showing the temporal sequence of frames.
    
    Args:
        frame_data: Dictionary of frame data
        output_path: Path to save the GIF
        colormap: Colormap for visualization
        max_frames: Maximum number of frames to include
    """
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    # Sort frames by frame number
    sorted_frames = sorted(frame_data.keys())
    
    # Limit to max_frames
    if len(sorted_frames) > max_frames:
        # Sample evenly
        indices = np.linspace(0, len(sorted_frames) - 1, max_frames, dtype=int)
        sorted_frames = [sorted_frames[i] for i in indices]
    
    # Create a frame for each time step
    for frame_num in sorted_frames:
        data = frame_data[frame_num]
        
        fig = plt.figure(figsize=(15, 8))
        
        # Ground truth mesh
        ax1 = fig.add_subplot(121, projection='3d')
        plot_mesh_with_contacts(ax1, data['gt_vertices'], data['edges'], data['gt_contacts'],
                               title=f"Ground Truth (Frame {frame_num})",
                               colormap=colormap)
        
        # Predicted mesh
        ax2 = fig.add_subplot(122, projection='3d')
        plot_mesh_with_contacts(ax2, data['pred_vertices'], data['edges'], data['pred_contacts'],
                               title=f"Predicted (Frame {frame_num})",
                               colormap=colormap)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2])
        cbar.set_label('Contact Probability')
        
        # Add metrics
        contact_accuracy = np.mean((data['pred_contacts'] > 0.5) == data['gt_contacts'])
        mean_error = np.mean(data['displacement_error'])
        
        metrics_text = f"Contact Accuracy: {contact_accuracy:.4f}, Mean Displacement Error: {mean_error:.4f}"
        fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{frame_num:05d}.png")
        plt.savefig(frame_path, dpi=100)
        frame_paths.append(frame_path)
        plt.close(fig)
    
    # Create GIF
    with imageio.get_writer(output_path, mode='I', duration=0.7) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)

def plot_mesh_with_contacts(ax, vertices, edges, values, title="", colormap=plt.cm.jet):
    """
    Plot a mesh with vertex coloring based on values.
    
    Args:
        ax: Matplotlib 3D axis
        vertices: Numpy array of vertex coordinates (n_vertices, 3)
        edges: List of edge tuples (src, dst)
        values: Numpy array of values to map to colors (n_vertices,)
        title: Plot title
        colormap: Matplotlib colormap
    """
    # Plot vertices
    scatter = ax.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        c=values, cmap=colormap, s=20, alpha=0.8
    )
    
    # Plot edges
    for src, dst in edges:
        ax.plot(
            [vertices[src, 0], vertices[dst, 0]],
            [vertices[src, 1], vertices[dst, 1]],
            [vertices[src, 2], vertices[dst, 2]],
            color='gray', linewidth=0.5, alpha=0.3
        )
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set consistent view angle
    ax.view_init(elev=30, azim=45)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    return scatter

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    
    # Set all defaults to match the training script
    parser.add_argument('--model_path', default='best_multi_task_model.pt', help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', default='/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0414_spam_pla/output', help='Directory containing test data')
    parser.add_argument('--output_dir', default='visualization_output', help='Directory to save visualizations')
    parser.add_argument('--frames', nargs='+', type=int, default=[8859], help='Specific frame numbers to visualize')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run model on')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--n_channels', type=int, default=5, help='Number of audio channels')
    parser.add_argument('--n_mels', type=int, default=64, help='Number of mel bins in spectrogram')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size in model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and data loader
    dataset = AudioMeshDataset(
        args.data_dir,
        n_mels=args.n_mels,
        remove_background=True
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load model
    model = AudioMeshMultiTaskModel(
        n_channels=args.n_channels, 
        n_mels=args.n_mels, 
        hidden_dim=args.hidden_dim
    )
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()
    
    # Run visualization
    visualize_results(
        model,
        data_loader,
        args.output_dir,
        args.data_dir,
        device=args.device,
        specific_frames=args.frames
    )
    
    print(f"Visualization complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()