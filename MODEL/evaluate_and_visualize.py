import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from model import AudioMeshDataset, AudioMeshContactModel, collate_fn
from torch.utils.data import DataLoader, Subset
from PIL import Image

def create_skip_frame_dataset(root_dir, skip_frames=100):
    """Create a dataset that only includes every nth frame."""
    full_dataset = AudioMeshDataset(root_dir)
    
    # Get indices for every nth frame
    indices = list(range(0, len(full_dataset), skip_frames))
    subset_dataset = Subset(full_dataset, indices)
    
    return subset_dataset

def visualize_mesh_with_contacts(vertices, pred_labels, true_labels, frame_num, data_root, output_path):
    """Create a visualization with original image and 3D mesh side by side."""
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(20, 10))
    
    # First subplot: Original image
    ax1 = fig.add_subplot(121)
    img_path = os.path.join(data_root, f'visualization_{frame_num:05d}.png')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Original Image')
    else:
        ax1.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        ax1.set_title(f'Missing: visualization_{frame_num:05d}.png')
    
    # Second subplot: 3D point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Normalize vertex positions for better visualization
    vertices = vertices - vertices.mean(axis=0)
    vertices = vertices / np.abs(vertices).max()
    
    # Create masks for different categories
    non_contact_mask = true_labels < 0.5  # Non-contact points (blue)
    correct_contact_mask = (true_labels >= 0.5) & (pred_labels >= 0.5)  # Correct contact predictions (green)
    incorrect_contact_mask = (true_labels >= 0.5) & (pred_labels < 0.5)  # Incorrect contact predictions (red)
    
    # Plot points with different colors
    ax2.scatter(vertices[non_contact_mask, 0], 
              vertices[non_contact_mask, 1], 
              vertices[non_contact_mask, 2], 
              c='blue', alpha=0.6, label='No Contact')
    
    ax2.scatter(vertices[correct_contact_mask, 0], 
              vertices[correct_contact_mask, 1], 
              vertices[correct_contact_mask, 2], 
              c='green', alpha=0.8, label='Correct Contact')
    
    ax2.scatter(vertices[incorrect_contact_mask, 0], 
              vertices[incorrect_contact_mask, 1], 
              vertices[incorrect_contact_mask, 2], 
              c='red', alpha=0.8, label='Incorrect Contact')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Set consistent viewing angle
    ax2.view_init(elev=30, azim=45)
    ax2.set_title('Contact Points Visualization\nBlue: No Contact, Green: Correct Contact, Red: Incorrect Contact')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_and_visualize(model_path, data_root, output_dir, device='cuda'):
    """Evaluate model and create visualizations for every nth frame."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = AudioMeshContactModel(n_channels=5, n_mels=64, hidden_dim=256)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = create_skip_frame_dataset(data_root, skip_frames=100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Lists to store metrics
    all_preds = []
    all_labels = []
    frame_images = []
    
    print(f"Evaluating and visualizing {len(dataset)} frames...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Get predictions
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            labels = mesh.y.cpu().numpy()
            
            logits = model(audio, mesh)
            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            
            # Store metrics
            all_preds.append(preds)
            all_labels.append(labels)
            
            # Create visualization
            frame_num = batch['frames'][0]
            output_path = os.path.join(output_dir, f'frame_{frame_num:05d}.png')
            
            vertices = mesh.x.cpu().numpy()
            visualize_mesh_with_contacts(vertices, preds, labels, frame_num, data_root, output_path)
            frame_images.append(imageio.imread(output_path))
            
            if i % 10 == 0:
                print(f"Processed frame {frame_num}")
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    precision = np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_preds == 1) + 1e-10)
    recall = np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_labels == 1) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create GIF
    gif_path = os.path.join(output_dir, 'contact_visualization.gif')
    imageio.mimsave(gif_path, frame_images, duration=0.5)  # 0.5 seconds per frame
    print(f"\nVisualization saved as: {gif_path}")

if __name__ == '__main__':
    # Configuration
    model_path = 'best_model.pt'  # Path to your trained model
    data_root = '/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0414_spam_pla/output'
    output_dir = 'evaluation_results'
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run evaluation and visualization
    evaluate_and_visualize(model_path, data_root, output_dir, device) 