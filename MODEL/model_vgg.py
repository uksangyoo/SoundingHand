import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import glob
import trimesh
import wandb
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GlobalAttention
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
import torchvision.models as models

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
                 remove_background=True):
        
        self.root_dir = root_dir
        self.transform = transform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.remove_background = remove_background
        audio_files = sorted(glob.glob(os.path.join(root_dir, audio_dir_pattern)))
        mesh_files = sorted(glob.glob(os.path.join(root_dir, mesh_dir_pattern)))
        label_files = sorted(glob.glob(os.path.join(root_dir, labels_pattern)))
        assert len(audio_files) > 0, f"No audio files found in {root_dir}/{audio_dir_pattern}"
        assert len(mesh_files) > 0, f"No mesh files found in {root_dir}/{mesh_dir_pattern}"
        assert len(label_files) > 0, f"No label files found in {root_dir}/{labels_pattern}"
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
        self.background_audio = None
        if self.remove_background and 0 in audio_dict:
            bg_path = audio_dict[0]
            waveform, sample_rate = torchaudio.load(bg_path)
            self.background_audio = waveform
            self.sample_rate = sample_rate
        sample_mesh = trimesh.load(self.frame_data[0]['mesh_path'])
        self.num_vertices = len(sample_mesh.vertices)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,  
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            normalized=True  # Normalize mel filterbanks
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0  # Clip values below -80 dB
        )

        # Pre-compute normalization stats from the first 100 samples (or all if less)
        print("Computing spectrogram normalization statistics...")
        n_samples = min(100, len(self.frame_data))
        temp_specs = []
        for i in range(n_samples):
            waveform, _ = torchaudio.load(self.frame_data[i]['audio_path'])
            if self.remove_background and self.background_audio is not None and waveform.shape == self.background_audio.shape:
                waveform = waveform - self.background_audio
            for channel in range(waveform.shape[0]):
                mel_spec = self.mel_transform(waveform[channel:channel+1])
                mel_spec = self.amplitude_to_db(mel_spec)
                temp_specs.append(mel_spec)
        temp_specs = torch.cat(temp_specs, dim=0)
        self.spec_mean = torch.mean(temp_specs)
        self.spec_std = torch.std(temp_specs)
        print(f"Spectrogram stats - Mean: {self.spec_mean:.2f}, Std: {self.spec_std:.2f}")

    def _get_frame_num(self, file_path):
        filename = os.path.basename(file_path)
        if 'frame_' in filename:
            if '_hand_' in filename:
                frame_str = filename.split('_hand_')[0].replace('frame_', '')
            elif 'contact_labels' in filename:
                frame_str = filename.split('_hand_')[0].replace('frame_', '')
            else:
                frame_str = filename.replace('frame_', '').split('.')[0]
            return int(frame_str)
        return -1

    def __len__(self):
        return len(self.frame_data)

    def normalize_spectrogram(self, spec):
        # Normalize frequency bins to sum to 1
        spec = F.normalize(spec, p=1, dim=1)
        # Standardize using pre-computed statistics
        spec = (spec - self.spec_mean) / (self.spec_std + 1e-8)
        return spec

    def __getitem__(self, idx):
        data = self.frame_data[idx]
        waveform, sample_rate = torchaudio.load(data['audio_path'])
        if self.remove_background and self.background_audio is not None and waveform.shape == self.background_audio.shape:
            waveform = waveform - self.background_audio
        specs = []
        for channel in range(waveform.shape[0]):
            channel_waveform = waveform[channel:channel+1]
            mel_spec = self.mel_transform(channel_waveform)
            mel_spec = self.amplitude_to_db(mel_spec)
            mel_spec = self.normalize_spectrogram(mel_spec)
            if mel_spec.size(-1) < 4:
                padding = torch.zeros(1, self.n_mels, 4 - mel_spec.size(-1), device=mel_spec.device)
                mel_spec = torch.cat([mel_spec, padding], dim=-1)
            specs.append(mel_spec)
        spectrogram = torch.stack(specs, dim=0)
        mesh = trimesh.load(data['mesh_path'])
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            edges = set()
            for face in mesh.faces:
                edges.add((face[0], face[1]))
                edges.add((face[1], face[0]))
                edges.add((face[1], face[2]))
                edges.add((face[2], face[1]))
                edges.add((face[2], face[0]))
                edges.add((face[0], face[2]))
            edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        else:
            from sklearn.neighbors import kneighbors_graph
            k = 6  
            adjacency = kneighbors_graph(mesh.vertices, k, mode='connectivity', include_self=False)
            edges = []
            for i, j in zip(*adjacency.nonzero()):
                edges.append((i, j))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        contact_labels = np.load(data['label_path'])
        contact_labels = torch.tensor(contact_labels, dtype=torch.float32)
        mesh_data = Data(
            x=vertices,
            edge_index=edge_index,
            y=contact_labels
        )
        return {
            'frame': data['frame'],
            'audio': spectrogram,
            'mesh': mesh_data
        }

class AudioEncoder(nn.Module):
    def __init__(self, in_channels=5, n_mels=64, hidden_dim=256, finetune_after_layer=7):
        super(AudioEncoder, self).__init__()
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=True)
        
        # Modify first conv layer to handle different input channels
        first_conv = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=(1, 1))
        # Initialize using the weights from the pretrained model's first layer
        with torch.no_grad():
            first_conv.weight[:, :3, :, :] = vgg16.features[0].weight[:, :3, :, :] / 3
            if in_channels > 3:
                first_conv.weight[:, 3:, :, :] = vgg16.features[0].weight[:, :3, :, :].mean(dim=1, keepdim=True).repeat(1, in_channels-3, 1, 1) / 3
        
        # Create new features sequence with our modified first conv
        self.features = nn.Sequential(
            first_conv,
            *list(vgg16.features[1:16])  # Use first few layers of VGG (up to conv4)
        )
        
        # Freeze layers before finetune_after_layer
        for i, param in enumerate(self.features.parameters()):
            if i < finetune_after_layer * 2:  # *2 because each layer has weights and bias
                param.requires_grad = False
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))
        
        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, n_mels, 4)  # Minimum spectrogram width
            dummy_output = self.adaptive_pool(self.features(dummy_input))
            flattened_size = dummy_output.numel()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(2)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        att = self.attention(x)
        x = x * att
        return x

class MeshEncoder(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=256, output_dim=256):
        super(MeshEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 128) 
        self.conv3 = GATConv(128, hidden_dim, heads=4, dropout=0.2, concat=False)
        self.global_attention = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        global_features = self.global_attention(x, batch=data.batch if hasattr(data, 'batch') else None)
        global_features = self.mlp(global_features)
        return global_features, x
    

class ContactPredictor(nn.Module):
    def __init__(self, audio_dim=256, mesh_dim=256, hidden_dim=256):
        super(ContactPredictor, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + mesh_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.vertex_attention = nn.Sequential(
            nn.Linear(mesh_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(mesh_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, audio_features, mesh_global_features, mesh_vertex_features):
        global_features = torch.cat([audio_features, mesh_global_features], dim=1)
        fused_features = self.fusion(global_features)
        batch_size = mesh_vertex_features.size(0) // fused_features.size(0)
        expanded_features = fused_features.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, fused_features.size(1))
        vertex_with_global = torch.cat([mesh_vertex_features, expanded_features], dim=1)
        attention_weights = self.vertex_attention(vertex_with_global)
        attended_features = vertex_with_global * attention_weights
        logits = self.mlp(attended_features)
        return logits.squeeze(-1)
    
class SoundingHandModel(nn.Module):

    def __init__(self, n_channels=5, n_mels=128, hidden_dim=256):
        super(SoundingHandModel, self).__init__()
        self.audio_encoder = AudioEncoder(in_channels=n_channels, n_mels=n_mels, hidden_dim=hidden_dim)
        self.mesh_encoder = MeshEncoder(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.contact_predictor = ContactPredictor(audio_dim=hidden_dim, mesh_dim=hidden_dim, hidden_dim=hidden_dim)
    
    def forward(self, audio, mesh_data):
        audio_features = self.audio_encoder(audio)
        mesh_global_features, mesh_vertex_features = self.mesh_encoder(mesh_data)
        contact_logits = self.contact_predictor(audio_features, mesh_global_features, mesh_vertex_features)
        return contact_logits
def collate_fn(batch):

    frames = [item['frame'] for item in batch]
    audio = torch.stack([item['audio'] for item in batch])
    mesh_data = Batch.from_data_list([item['mesh'] for item in batch])
    return {
        'frames': frames,
        'audio': audio,
        'mesh': mesh_data
    }

def create_data_loaders(root_dir, batch_size=8, num_workers=4):

    dataset = AudioMeshDataset(root_dir)
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

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    os.makedirs('checkpoints', exist_ok=True)
    wandb.init(project="sounding-hand", name="contact-prediction-training")
    wandb.config.update({
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "model_hidden_dim": model.audio_encoder.fc[1].in_features,
        "optimizer": "Adam"
    })

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            labels = mesh.y.to(device)
            optimizer.zero_grad()
            logits = model(audio, mesh)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                mesh = batch['mesh'].to(device)
                labels = mesh.y.to(device)
                logits = model(audio, mesh)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join('checkpoints', 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            
            wandb.save(best_model_path)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
    wandb.finish()
    return model


def evaluate_model(model, test_loader, device='cuda'):

    model = model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0
    all_preds = []
    all_labels = []


    with torch.no_grad():

        for batch in test_loader:
            audio = batch['audio'].to(device)
            mesh = batch['mesh'].to(device)
            labels = mesh.y.to(device)
            logits = model(audio, mesh)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())


    avg_test_loss = test_loss / len(test_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    precision = np.sum((all_preds == 1) & (all_labels == 1)) / np.sum(all_preds == 1)
    recall = np.sum((all_preds == 1) & (all_labels == 1)) / np.sum(all_labels == 1)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
    return avg_test_loss, accuracy, f1



if __name__ == '__main__':

    root_dir = '/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0414_spam_pla/output'
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(root_dir, batch_size=4)
    print("Sample batch:")


    for batch in train_loader:
        print(batch['frames'])
        print(batch['audio'].shape)
        print(batch['mesh'])
        break


    print("Creating model...")
    model = SoundingHandModel(n_channels=5, n_mels=128, hidden_dim=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print("Training model...")


    trained_model = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
    print("Evaluating model...")
    evaluate_model(trained_model, val_loader, device=device)