import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import glob
import trimesh
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GlobalAttention
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import imageio
import os
import tempfile
def collate_fn(batch):
    """Custom collate function for batching heterogeneous data."""
    frames = [item['frame'] for item in batch]
    audio = torch.stack([item['audio'] for item in batch])
    
    # Process mesh data
    mesh_data = Batch.from_data_list([item['mesh'] for item in batch])
    
    return {
        'frames': frames,
        'audio': audio,
        'mesh': mesh_data
    }

def create_data_loaders(root_dir, batch_size=8, num_workers=4):
    """Create train and validation data loaders."""
    dataset = AudioMeshDataset(root_dir)
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
# Dataset class can remain mostly the same, just need to add reference mesh to compute displacements
class AudioMeshDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 audio_dir_pattern="audio/frame_*.wav", 
                 mesh_dir_pattern="hands/frame_*_hand_1.obj",
                 labels_pattern="hands/frame_*_hand_contact_labels.npy",
                 transform=None,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 remove_background=True,
                 reference_mesh_path=None):  # Add reference mesh path
        """
        Dataset for loading audio and mesh data with contact labels.
        
        Args:
            root_dir: Root directory containing subdirectories with data
            audio_dir_pattern: Pattern to match audio files
            mesh_dir_pattern: Pattern to match mesh files
            labels_pattern: Pattern to match label files
            transform: Optional transform to apply to the data
            n_fft: FFT window size for spectrogram
            hop_length: Hop length for spectrogram
            n_mels: Number of mel bands
            remove_background: Whether to subtract the first frame as background
            reference_mesh_path: Path to reference mesh for computing displacements (if None, use first frame)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.remove_background = remove_background
        
        # Find all frame numbers by parsing filenames
        audio_files = sorted(glob.glob(os.path.join(root_dir, audio_dir_pattern)))
        mesh_files = sorted(glob.glob(os.path.join(root_dir, mesh_dir_pattern)))
        label_files = sorted(glob.glob(os.path.join(root_dir, labels_pattern)))
        
        # Ensure we have matching files
        assert len(audio_files) > 0, f"No audio files found in {root_dir}/{audio_dir_pattern}"
        assert len(mesh_files) > 0, f"No mesh files found in {root_dir}/{mesh_dir_pattern}"
        assert len(label_files) > 0, f"No label files found in {root_dir}/{labels_pattern}"
        
        # Match files by frame number
        self.frame_data = []
        audio_dict = {self._get_frame_num(f): f for f in audio_files}
        mesh_dict = {self._get_frame_num(f): f for f in mesh_files}
        label_dict = {self._get_frame_num(f): f for f in label_files}
        
        common_frames = set(audio_dict.keys()) & set(mesh_dict.keys()) & set(label_dict.keys())
        
        for frame in sorted(common_frames):
            self.frame_data.append({
                'frame': frame,
                'audio_path': audio_dict[frame],
                'mesh_path': mesh_dict[frame],
                'label_path': label_dict[frame]
            })
        
        # Get background audio (frame 0) for background subtraction
        self.background_audio = None
        if self.remove_background and 0 in audio_dict:
            bg_path = audio_dict[0]
            waveform, sample_rate = torchaudio.load(bg_path)
            self.background_audio = waveform
            self.sample_rate = sample_rate
            
        # Load a sample mesh to get the number of vertices
        sample_mesh = trimesh.load(self.frame_data[0]['mesh_path'])
        self.num_vertices = len(sample_mesh.vertices)
        
        # Set reference mesh for computing displacements
        self.reference_mesh = None
        if reference_mesh_path:
            self.reference_mesh = trimesh.load(reference_mesh_path)
        else:
            # Use first frame as reference if no specific reference provided
            self.reference_mesh = sample_mesh
            
        self.ref_vertices = torch.tensor(self.reference_mesh.vertices, dtype=torch.float32)
        
        # Create a mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Log amplitude
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
    def _get_frame_num(self, file_path):
        """Extract frame number from file path."""
        filename = os.path.basename(file_path)
        if 'frame_' in filename:
            # Handle different filename formats
            if '_hand_' in filename:
                # For mesh files: frame_0_hand_1.obj
                frame_str = filename.split('_hand_')[0].replace('frame_', '')
            elif 'contact_labels' in filename:
                # For label files: frame_0_hand_contact_labels.npy
                frame_str = filename.split('_hand_')[0].replace('frame_', '')
            else:
                # For audio files: frame_00000.wav
                frame_str = filename.replace('frame_', '').split('.')[0]
                
            # Convert to integer, handling both zero-padded and non-padded formats
            return int(frame_str)
        return -1
        
    def __len__(self):
        return len(self.frame_data)
    
    def __getitem__(self, idx):
        data = self.frame_data[idx]
        
        # Load audio and convert to mel spectrogram
        waveform, sample_rate = torchaudio.load(data['audio_path'])
        
        # Background subtraction if enabled
        if self.remove_background and self.background_audio is not None and waveform.shape == self.background_audio.shape:
            waveform = waveform - self.background_audio
            
        # Multi-channel audio (5 channels)
        # Compute spectrograms for each channel
        specs = []
        for channel in range(waveform.shape[0]):
            channel_waveform = waveform[channel:channel+1]
            mel_spec = self.mel_transform(channel_waveform)
            mel_spec = self.amplitude_to_db(mel_spec)
            
            # Add padding to time dimension if too small (ensure at least 4 time steps)
            if mel_spec.size(-1) < 4:
                padding = torch.zeros(1, self.n_mels, 4 - mel_spec.size(-1), device=mel_spec.device)
                mel_spec = torch.cat([mel_spec, padding], dim=-1)
                
            specs.append(mel_spec)
        
        # Stack spectrograms along the channel dimension [channels, mels, time]
        spectrogram = torch.stack(specs, dim=0)
        
        # Load mesh vertices
        mesh = trimesh.load(data['mesh_path'])
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        
        # Compute displacements from reference mesh
        displacements = vertices - self.ref_vertices
        
        # Create edge indices based on mesh faces
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Extract edges from faces
            edges = set()
            for face in mesh.faces:
                # Add all edges in the face (both directions for undirected graph)
                edges.add((face[0], face[1]))
                edges.add((face[1], face[0]))
                edges.add((face[1], face[2]))
                edges.add((face[2], face[1]))
                edges.add((face[2], face[0]))
                edges.add((face[0], face[2]))
            
            # Convert to tensor
            edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        else:
            # Fallback: connect each vertex to its k nearest neighbors
            from sklearn.neighbors import kneighbors_graph
            k = 6  # Number of neighbors
            adjacency = kneighbors_graph(mesh.vertices, k, mode='connectivity', include_self=False)
            edges = []
            for i, j in zip(*adjacency.nonzero()):
                edges.append((i, j))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Load contact labels
        contact_labels = np.load(data['label_path'])
        contact_labels = torch.tensor(contact_labels, dtype=torch.float32)
        
        # Create PyG data object with both vertex positions and displacements
        mesh_data = Data(
            x=vertices,  # Original vertices
            edge_index=edge_index,
            y=contact_labels,
            displacements=displacements,  # Store displacements as target
            ref_vertices=self.ref_vertices  # Store reference vertices
        )
        
        return {
            'frame': data['frame'],
            'audio': spectrogram,
            'mesh': mesh_data
        }

# AudioEncoder can remain the same
class AudioEncoder(nn.Module):
    def __init__(self, in_channels=5, n_mels=64, hidden_dim=256):
        super(AudioEncoder, self).__init__()
        
        # First layer: process along mel dimension only for small time dimensions
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1)),  # Only pool along mel dimension
            nn.Dropout(0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1)),  # Only pool along mel dimension
            nn.Dropout(0.2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 2)),  # Pool in both dimensions for final reduction
            nn.Dropout(0.3)
        )
        
        # Use adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))
        
        # Calculate the output size after adaptive pooling
        pooled_size = 128 * 4 * 2
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input: [batch, channels, mels, time] or [batch, channels, 1, mels, time]
        # Check if we need to reshape the input from 5D to 4D
        if x.dim() == 5:
            # Reshape from [batch, channels, 1, mels, time] to [batch, channels, mels, time]
            x = x.squeeze(2)
            
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        att = self.attention(x)
        x = x * att
        return x

# Enhance MeshEncoder to better capture spatial relationships
class EnhancedMeshEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=256):
        super(EnhancedMeshEncoder, self).__init__()
        
        # Graph convolutional layers with residual connections
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        
        # Multi-head attention for better feature extraction
        self.conv3 = GATConv(128, hidden_dim // 4, heads=4, dropout=0.2)
        
        # Edge features processing
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * (hidden_dim // 4) * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Global attention for graph-level features
        self.global_attention = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        # MLP for final encoding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Layer norms for stable training
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(128)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply graph convolutions with residual connections and layer normalization
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.layer_norm1(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.layer_norm2(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        # Multi-head attention
        x3 = self.conv3(x2, edge_index)
        x3 = self.layer_norm3(x3)
        
        # Extract graph-level representation
        global_features = self.global_attention(x3, batch=data.batch if hasattr(data, 'batch') else None)
        
        # Apply MLP
        global_features = self.mlp(global_features)
        
        return global_features, x3

# New class for mesh displacement prediction
class MeshDisplacementPredictor(nn.Module):
    def __init__(self, audio_dim=256, mesh_dim=256, hidden_dim=256):
        super(MeshDisplacementPredictor, self).__init__()
        
        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + mesh_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Hierarchical vertex feature processing
        self.vertex_processor = nn.Sequential(
            nn.Linear(mesh_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Separate prediction heads for each coordinate dimension
        # This allows the model to learn different patterns for each dimension
        self.x_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.y_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.z_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, audio_features, mesh_global_features, mesh_vertex_features):
        # Concatenate global audio and mesh features
        global_features = torch.cat([audio_features, mesh_global_features], dim=1)
        
        # Apply fusion MLP
        fused_features = self.fusion(global_features)
        
        # Expand fused features to match the number of vertices
        batch_size = mesh_vertex_features.size(0) // fused_features.size(0)
        expanded_features = fused_features.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, fused_features.size(1))
        
        # Concatenate vertex features with fused global features
        vertex_with_global = torch.cat([mesh_vertex_features, expanded_features], dim=1)
        
        # Process vertex features
        processed_features = self.vertex_processor(vertex_with_global)
        
        # Predict displacements for each coordinate separately
        x_displacement = self.x_predictor(processed_features)
        y_displacement = self.y_predictor(processed_features)
        z_displacement = self.z_predictor(processed_features)
        
        # Combine predictions [num_vertices, 3]
        displacements = torch.cat([x_displacement, y_displacement, z_displacement], dim=1)
        
        return displacements

# Contact predictor remains similar but with some enhancements
class EnhancedContactPredictor(nn.Module):
    def __init__(self, audio_dim=256, mesh_dim=256, hidden_dim=256):
        super(EnhancedContactPredictor, self).__init__()
        
        # Cross-modal fusion with residual connection
        self.fusion1 = nn.Linear(audio_dim + mesh_dim, hidden_dim)
        self.fusion2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention mechanism for vertex features
        self.vertex_attention = nn.Sequential(
            nn.Linear(mesh_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # MLP for final prediction with deeper network
        self.mlp = nn.Sequential(
            nn.Linear(mesh_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Layer norms for stable training
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio_features, mesh_global_features, mesh_vertex_features):
        # Concatenate global audio and mesh features
        global_features = torch.cat([audio_features, mesh_global_features], dim=1)
        
        # Apply fusion with residual connection
        fused_features = F.relu(self.fusion1(global_features))
        fused_features = self.layer_norm1(fused_features)
        fused_features = F.dropout(fused_features, p=0.3, training=self.training)
        residual = fused_features
        
        fused_features = F.relu(self.fusion2(fused_features))
        fused_features = self.layer_norm2(fused_features)
        fused_features = F.dropout(fused_features, p=0.3, training=self.training)
        fused_features = fused_features + residual  # Residual connection
        
        # Expand fused features to match the number of vertices
        batch_size = mesh_vertex_features.size(0) // fused_features.size(0)
        expanded_features = fused_features.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, fused_features.size(1))
        
        # Concatenate vertex features with fused global features
        vertex_with_global = torch.cat([mesh_vertex_features, expanded_features], dim=1)
        
        # Apply attention to focus on important vertices
        attention_weights = self.vertex_attention(vertex_with_global)
        attended_features = vertex_with_global * attention_weights
        
        # Final prediction
        logits = self.mlp(attended_features)
        
        return logits.squeeze(-1)

# New multi-task model
class AudioMeshMultiTaskModel(nn.Module):
    def __init__(self, n_channels=5, n_mels=128, hidden_dim=256):
        super(AudioMeshMultiTaskModel, self).__init__()
        
        self.audio_encoder = AudioEncoder(in_channels=n_channels, n_mels=n_mels, hidden_dim=hidden_dim)
        self.mesh_encoder = EnhancedMeshEncoder(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Two separate predictors
        self.contact_predictor = EnhancedContactPredictor(audio_dim=hidden_dim, mesh_dim=hidden_dim, hidden_dim=hidden_dim)
        self.displacement_predictor = MeshDisplacementPredictor(audio_dim=hidden_dim, mesh_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # Task weighting parameters (learnable)
        self.log_var_contact = nn.Parameter(torch.zeros(1))
        self.log_var_displacement = nn.Parameter(torch.zeros(1))
        
    def forward(self, audio, mesh_data):
        # Process audio
        audio_features = self.audio_encoder(audio)
        
        # Process mesh
        mesh_global_features, mesh_vertex_features = self.mesh_encoder(mesh_data)
        
        # Predict contact
        contact_logits = self.contact_predictor(audio_features, mesh_global_features, mesh_vertex_features)
        
        # Predict displacements
        displacements = self.displacement_predictor(audio_features, mesh_global_features, mesh_vertex_features)
        
        return contact_logits, displacements
    
    def get_task_weights(self):
        # Returns task weights for loss balancing using uncertainty weighting
        # See: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
        precision_contact = torch.exp(-self.log_var_contact)
        precision_displacement = torch.exp(-self.log_var_displacement)
        
        return {
            'contact': precision_contact,
            'displacement': precision_displacement,
            'contact_var': self.log_var_contact,
            'displacement_var': self.log_var_displacement
        }

# Modify collate function to include displacements
def collate_fn(batch):
    """Custom collate function for batching heterogeneous data."""
    frames = [item['frame'] for item in batch]
    audio = torch.stack([item['audio'] for item in batch])
    
    # Process mesh data
    mesh_data = Batch.from_data_list([item['mesh'] for item in batch])
    
    return {
        'frames': frames,
        'audio': audio,
        'mesh': mesh_data
    }

# Modify training function for multi-task learning
def train_multi_task_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """Train the multi-task model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss functions
    contact_criterion = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to apply per-sample weights later
    displacement_criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply per-sample weights later
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_contact_loss = 0
        train_displacement_loss = 0
        
        for batch in train_loader:
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            contact_labels = mesh.y.to(device)
            displacement_labels = mesh.displacements.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            contact_logits, predicted_displacements = model(audio, mesh)
            
            # Get task weights
            task_weights = model.get_task_weights()
            
            # Calculate losses
            contact_loss_raw = contact_criterion(contact_logits, contact_labels)
            displacement_loss_raw = displacement_criterion(predicted_displacements, displacement_labels)
            
            # Apply uncertainty weighting
            contact_loss = torch.mean(task_weights['contact'] * contact_loss_raw) + task_weights['contact_var']
            displacement_loss = torch.mean(task_weights['displacement'] * torch.mean(displacement_loss_raw, dim=1)) + task_weights['displacement_var']
            
            # Combined loss
            loss = contact_loss + displacement_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_contact_loss += contact_loss.item()
            train_displacement_loss += displacement_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_contact_loss = train_contact_loss / len(train_loader)
        avg_train_displacement_loss = train_displacement_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_contact_loss = 0
        val_displacement_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                mesh = batch['mesh'].to(device)
                contact_labels = mesh.y.to(device)
                displacement_labels = mesh.displacements.to(device)
                
                # Forward pass
                contact_logits, predicted_displacements = model(audio, mesh)
                
                # Get task weights
                task_weights = model.get_task_weights()
                
                # Calculate losses
                contact_loss_raw = contact_criterion(contact_logits, contact_labels)
                displacement_loss_raw = displacement_criterion(predicted_displacements, displacement_labels)
                
                # Apply uncertainty weighting
                contact_loss = torch.mean(task_weights['contact'] * contact_loss_raw) + task_weights['contact_var']
                displacement_loss = torch.mean(task_weights['displacement'] * torch.mean(displacement_loss_raw, dim=1)) + task_weights['displacement_var']
                
                # Combined loss
                loss = contact_loss + displacement_loss
                
                val_loss += loss.item()
                val_contact_loss += contact_loss.item()
                val_displacement_loss += displacement_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_contact_loss = val_contact_loss / len(val_loader)
        avg_val_displacement_loss = val_displacement_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} (Contact: {avg_train_contact_loss:.4f}, Displacement: {avg_train_displacement_loss:.4f})')
        print(f'Val Loss: {avg_val_loss:.4f} (Contact: {avg_val_contact_loss:.4f}, Displacement: {avg_val_displacement_loss:.4f})')
        print(f'Task weights: Contact={task_weights["contact"].item():.4f}, Displacement={task_weights["displacement"].item():.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_multi_task_model.pt')
    
    return model

def evaluate_multi_task_model(model, test_loader, device='cuda'):
    """Evaluate the multi-task model."""
    model = model.to(device)
    model.eval()
    
    contact_criterion = nn.BCEWithLogitsLoss()
    displacement_criterion = nn.MSELoss()
    
    test_loss = 0
    contact_loss = 0
    displacement_loss = 0
    
    all_contact_preds = []
    all_contact_labels = []
    all_displacement_preds = []
    all_displacement_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            contact_labels = mesh.y.to(device)
            displacement_labels = mesh.displacements.to(device)
            
            # Forward pass
            contact_logits, predicted_displacements = model(audio, mesh)
            
            # Calculate losses
            c_loss = contact_criterion(contact_logits, contact_labels)
            d_loss = displacement_criterion(predicted_displacements, displacement_labels)
            
            # Total loss (simple sum for evaluation)
            loss = c_loss + d_loss
            
            contact_loss += c_loss.item()
            displacement_loss += d_loss.item()
            test_loss += loss.item()
            
            # Store predictions
            contact_preds = torch.sigmoid(contact_logits) > 0.5
            all_contact_preds.append(contact_preds.cpu().numpy())
            all_contact_labels.append(contact_labels.cpu().numpy())
            
            all_displacement_preds.append(predicted_displacements.cpu().numpy())
            all_displacement_labels.append(displacement_labels.cpu().numpy())
    
    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_contact_loss = contact_loss / len(test_loader)
    avg_displacement_loss = displacement_loss / len(test_loader)
    
    # Contact metrics
    all_contact_preds = np.concatenate(all_contact_preds)
    all_contact_labels = np.concatenate(all_contact_labels)
    
    contact_accuracy = np.mean(all_contact_preds == all_contact_labels)
    contact_precision = np.sum((all_contact_preds == 1) & (all_contact_labels == 1)) / (np.sum(all_contact_preds == 1) + 1e-8)
    contact_recall = np.sum((all_contact_preds == 1) & (all_contact_labels == 1)) / (np.sum(all_contact_labels == 1) + 1e-8)
    contact_f1 = 2 * contact_precision * contact_recall / (contact_precision + contact_recall + 1e-8)
    
    # Displacement metrics
    all_displacement_preds = np.concatenate(all_displacement_preds)
    all_displacement_labels = np.concatenate(all_displacement_labels)
    
    displacement_mse = np.mean((all_displacement_preds - all_displacement_labels) ** 2)
    displacement_mae = np.mean(np.abs(all_displacement_preds - all_displacement_labels))
    
    print(f'Test Loss: {avg_test_loss:.4f} (Contact: {avg_contact_loss:.4f}, Displacement: {avg_displacement_loss:.4f})')
    print(f'Contact Metrics - Accuracy: {contact_accuracy:.4f}, F1: {contact_f1:.4f}')
    print(f'Contact Metrics - Precision: {contact_precision:.4f}, Recall: {contact_recall:.4f}')
    print(f'Displacement Metrics - MSE: {displacement_mse:.4f}, MAE: {displacement_mae:.4f}')
    
    return {
        'avg_test_loss': avg_test_loss,
        'contact': {
            'accuracy': contact_accuracy,
            'f1': contact_f1,
            'precision': contact_precision,
            'recall': contact_recall
        },
        'displacement': {
            'mse': displacement_mse,
            'mae': displacement_mae
        }
    }

# Modified visualization function for both contact and displacement predictions
def visualize_predictions(model, sample_batch, device='cuda'):
    """Visualize model predictions for a sample batch."""
    model.eval()
    
    audio = sample_batch['audio'].to(device)
    mesh = sample_batch['mesh'].to(device)
    contact_labels = mesh.y.cpu().numpy()
    displacement_labels = mesh.displacements.cpu().numpy()
    
    with torch.no_grad():
        contact_logits, predicted_displacements = model(audio, mesh)
        contact_probs = torch.sigmoid(contact_logits).cpu().numpy()
        predicted_displacements = predicted_displacements.cpu().numpy()
    
    # Visualize contact predictions (first sample in batch)
    sample_idx = 0
    frame_num = sample_batch['frames'][sample_idx]
    
    # Get number of vertices in the first sample
    vertices_per_sample = len(contact_labels) // len(sample_batch['frames'])
    start_idx = sample_idx * vertices_per_sample
    end_idx = start_idx + vertices_per_sample
    
    sample_contact_labels = contact_labels[start_idx:end_idx]
    sample_contact_probs = contact_probs[start_idx:end_idx]
    sample_displacements_labels = displacement_labels[start_idx:end_idx]
    sample_displacements_preds = predicted_displacements[start_idx:end_idx]
    
    # Compute displacement errors
    displacement_errors = np.sqrt(np.sum((sample_displacements_preds - sample_displacements_labels) ** 2, axis=1))
    
    # Get reference vertices to visualize predicted mesh
    ref_vertices = mesh.ref_vertices[start_idx:end_idx].cpu().numpy()
    predicted_vertices = ref_vertices + sample_displacements_preds
    
    print(f"Frame {frame_num} visualization statistics:")
    print(f"Contact - True positive rate: {np.mean(sample_contact_labels == 1):.4f}")
    print(f"Contact - Prediction confidence: {np.mean(sample_contact_probs):.4f}")
    print(f"Displacement - Mean error: {np.mean(displacement_errors):.4f}")
    print(f"Displacement - Max error: {np.max(displacement_errors):.4f}")
    
    # This would typically save the visualization to a file
    # But for simplicity, we'll just print some statistics
    print(f"Would save visualization for frame {frame_num} with {vertices_per_sample} vertices")
    print(f"Contact prediction range: {np.min(sample_contact_probs):.4f} - {np.max(sample_contact_probs):.4f}")
    print(f"Displacement prediction range: {np.min(np.linalg.norm(sample_displacements_preds, axis=1)):.4f} - {np.max(np.linalg.norm(sample_displacements_preds, axis=1)):.4f}")
    
    return {
        'frame': frame_num,
        'contact': {
            'labels': sample_contact_labels,
            'predictions': sample_contact_probs
        },
        'displacement': {
            'labels': sample_displacements_labels,
            'predictions': sample_displacements_preds,
            'errors': displacement_errors
        },
        'vertices': {
            'reference': ref_vertices,
            'predicted': predicted_vertices
        }
    }


def visualize_predictions_gif(model, test_loader, output_path, device='cuda', num_frames=30):
    """
    Visualize model predictions as GIF animation comparing predictions with ground truth.
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        output_path: Path to save the output GIF
        device: Device to run the model on
        num_frames: Maximum number of frames to include in the GIF
    """
    model.eval()
    
    # Collect samples
    samples = []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            
            # Get predictions
            contact_logits, predicted_displacements = model(audio, mesh)
            contact_probs = torch.sigmoid(contact_logits)
            
            # Process each sample in the batch
            vertices_per_sample = mesh.x.size(0) // len(batch['frames'])
            for i, frame_num in enumerate(batch['frames']):
                # Extract data for this sample
                start_idx = i * vertices_per_sample
                end_idx = start_idx + vertices_per_sample
                
                # Get vertices and labels
                ref_vertices = mesh.ref_vertices[start_idx:end_idx].cpu().numpy()
                gt_vertices = mesh.x[start_idx:end_idx].cpu().numpy()
                gt_displacements = mesh.displacements[start_idx:end_idx].cpu().numpy()
                pred_displacements = predicted_displacements[start_idx:end_idx].cpu().numpy()
                pred_vertices = ref_vertices + pred_displacements
                
                # Get contact probabilities and labels
                gt_contacts = mesh.y[start_idx:end_idx].cpu().numpy()
                pred_contacts = contact_probs[start_idx:end_idx].cpu().numpy()
                
                # Get edge indices for this sample (need to adjust indices to start from 0)
                edge_index = mesh.edge_index.cpu().numpy()
                # Filter edges for this sample and adjust indices
                sample_edges = []
                for j in range(edge_index.shape[1]):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    if start_idx <= src < end_idx and start_idx <= dst < end_idx:
                        sample_edges.append((src - start_idx, dst - start_idx))
                
                samples.append({
                    'frame': frame_num,
                    'ref_vertices': ref_vertices,
                    'gt_vertices': gt_vertices,
                    'pred_vertices': pred_vertices,
                    'gt_contacts': gt_contacts,
                    'pred_contacts': pred_contacts,
                    'edges': sample_edges
                })
            
            if len(samples) >= num_frames:
                break
    
    # Sort samples by frame number
    samples.sort(key=lambda x: x['frame'])
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    # Create custom color maps
    contact_cmap = LinearSegmentedColormap.from_list('contact', [(0, 'blue'), (0.5, 'lightblue'), (1, 'red')])
    error_cmap = plt.cm.jet
    
    # Create and save each frame
    frame_paths = []
    for i, sample in enumerate(samples):
        fig = plt.figure(figsize=(18, 6))
        
        # Plot ground truth mesh with contact points
        ax1 = fig.add_subplot(131, projection='3d')
        plot_mesh_with_contacts(ax1, sample['gt_vertices'], sample['edges'], sample['gt_contacts'],
                               title=f"Ground Truth (Frame {sample['frame']})",
                               colormap=contact_cmap)
        
        # Plot predicted mesh with contact probabilities
        ax2 = fig.add_subplot(132, projection='3d')
        plot_mesh_with_contacts(ax2, sample['pred_vertices'], sample['edges'], sample['pred_contacts'],
                               title="Predicted",
                               colormap=contact_cmap)
        
        # Plot error visualization
        ax3 = fig.add_subplot(133, projection='3d')
        displacement_errors = np.sqrt(np.sum((sample['pred_vertices'] - sample['gt_vertices'])**2, axis=1))
        max_error = displacement_errors.max()
        normalized_errors = displacement_errors / max_error if max_error > 0 else displacement_errors
        plot_mesh_with_contacts(ax3, sample['pred_vertices'], sample['edges'], normalized_errors,
                               title=f"Error (Max: {max_error:.4f})",
                               colormap=error_cmap)
        
        # Add colorbar for error
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=error_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Normalized Error')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        frame_paths.append(frame_path)
        plt.close(fig)
    
    # Create GIF
    with imageio.get_writer(output_path, mode='I', duration=0.2) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"GIF saved to {output_path}")

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
    
    # Make axis limits consistent
    ax.set_box_aspect([1, 1, 1])
    
    return scatter

def visualize_frames_sequence(model, test_loader, output_path, device='cuda', num_frames=5):
    """
    Visualize a sequence of frames showing both the model predictions and ground truth.
    Creates a single image with rows of frames showing the progression over time.
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        output_path: Path to save the output image
        device: Device to run the model on
        num_frames: Number of frames to include in the sequence
    """
    model.eval()
    
    # Collect consecutive frames
    frame_data = []
    current_frames = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            
            # Get predictions
            contact_logits, predicted_displacements = model(audio, mesh)
            contact_probs = torch.sigmoid(contact_logits)
            
            # Process each sample in the batch
            vertices_per_sample = mesh.x.size(0) // len(batch['frames'])
            for i, frame_num in enumerate(batch['frames']):
                if frame_num in current_frames:
                    continue
                    
                # Extract data for this sample
                start_idx = i * vertices_per_sample
                end_idx = start_idx + vertices_per_sample
                
                # Get vertices and labels
                gt_vertices = mesh.x[start_idx:end_idx].cpu().numpy()
                ref_vertices = mesh.ref_vertices[start_idx:end_idx].cpu().numpy()
                pred_displacements = predicted_displacements[start_idx:end_idx].cpu().numpy()
                pred_vertices = ref_vertices + pred_displacements
                
                # Get contact probabilities and labels
                gt_contacts = mesh.y[start_idx:end_idx].cpu().numpy()
                pred_contacts = contact_probs[start_idx:end_idx].cpu().numpy()
                
                # Get edges for this sample
                edge_index = mesh.edge_index.cpu().numpy()
                sample_edges = []
                for j in range(edge_index.shape[1]):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    if start_idx <= src < end_idx and start_idx <= dst < end_idx:
                        sample_edges.append((src - start_idx, dst - start_idx))
                
                frame_data.append({
                    'frame': frame_num,
                    'gt_vertices': gt_vertices,
                    'pred_vertices': pred_vertices,
                    'gt_contacts': gt_contacts,
                    'pred_contacts': pred_contacts,
                    'edges': sample_edges
                })
                
                current_frames.append(frame_num)
                
                if len(current_frames) >= num_frames:
                    break
            
            if len(current_frames) >= num_frames:
                break
    
    # Sort frames by frame number
    frame_data.sort(key=lambda x: x['frame'])
    
    # Create plot
    fig = plt.figure(figsize=(num_frames * 4, 8))
    
    # Custom colormap
    contact_cmap = LinearSegmentedColormap.from_list('contact', [(0, 'blue'), (0.5, 'lightblue'), (1, 'red')])
    
    # Plot ground truth row
    for i, data in enumerate(frame_data):
        ax = fig.add_subplot(2, num_frames, i + 1, projection='3d')
        plot_mesh_with_contacts(ax, data['gt_vertices'], data['edges'], data['gt_contacts'],
                              title=f"Ground Truth (Frame {data['frame']})",
                              colormap=contact_cmap)
    
    # Plot prediction row
    for i, data in enumerate(frame_data):
        ax = fig.add_subplot(2, num_frames, num_frames + i + 1, projection='3d')
        plot_mesh_with_contacts(ax, data['pred_vertices'], data['edges'], data['pred_contacts'],
                              title=f"Predicted (Frame {data['frame']})",
                              colormap=contact_cmap)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    sm = plt.cm.ScalarMappable(cmap=contact_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Contact Probability')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Sequence visualization saved to {output_path}")

if __name__ == '__main__':
    # Example usage
    root_dir = '/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0414_spam_pla/output'
    
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(root_dir, batch_size=4)
    
    print("Sample batch:")
    for batch in train_loader:
        print(batch['frames'])
        print(batch['audio'].shape)
        print(batch['mesh'])
        sample_batch = batch
        break
    
    print("Creating multi-task model...")
    model = AudioMeshMultiTaskModel(n_channels=5, n_mels=64, hidden_dim=256)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Training multi-task model...")
    trained_model = train_multi_task_model(model, train_loader, val_loader, num_epochs=30, device=device)
    
    print("Evaluating multi-task model...")
    metrics = evaluate_multi_task_model(trained_model, val_loader, device=device)
    
    print("Visualizing predictions...")
    vis_data = visualize_predictions(trained_model, sample_batch, device=device)

    # Create GIF visualization
    visualize_predictions_gif(model, val_loader, "hand_predictions.gif", device=device, num_frames=20)
    
    # Create sequence visualization
    visualize_frames_sequence(model, val_loader, "hand_sequence.png", device=device, num_frames=5)