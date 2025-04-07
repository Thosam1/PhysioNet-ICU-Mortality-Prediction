import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.mps.is_available() else
                      "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=49):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ContrastiveICUTransformer(nn.Module):
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_layers=5, dim_feedforward=256, dropout=0.1, projection_dim=32):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)

        # Embed input
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)

        # Permute for transformer: (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (seq_len, batch_size, d_model)

        # Take the mean of the sequence dimension to get a fixed-size embedding
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)

        # Pass through the projection head
        x = self.projection_head(x)  # Shape: (batch_size, projection_dim)
        return x

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        # Normalize embeddings
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(z_i, z_j.t()) / self.temperature

        # Create labels for positive pairs
        batch_size = z_i.size(0)
        labels = torch.arange(batch_size).to(z_i.device)

        # Compute loss
        loss = self.criterion(similarity_matrix, labels)
        return loss

# Example training loop
def train_contrastive_model(model, dataloader, optimizer, loss_fn, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Assume batch contains two augmented views of the same data
            x_i, x_j = batch
            x_i, x_j = x_i.to(device), x_j.to(device)

            # Forward pass
            z_i = model(x_i)
            z_j = model(x_j)

            # Compute loss
            loss = loss_fn(z_i, z_j)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

# Example dataset and dataloader using X_train_lstm
class AugmentedDataset(Dataset):
    def __init__(self, data, augment_fn):
        self.data = data
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x_i = self.augment_fn(x)  # Augmented view 1
        x_j = self.augment_fn(x)  # Augmented view 2
        return x_i, x_j

# Define augmentation function (example: adding noise)
def augment_fn(x):
    # Add noise
    noise = torch.randn_like(x) * 0.1
    x_noisy = x + noise

    # Time masking
    seq_len = x_noisy.size(0)
    mask_len = int(seq_len * 0.1)  # Mask 10% of the sequence
    mask_start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
    x_noisy[mask_start:mask_start + mask_len] = 0  # Mask with zeros

    return x_noisy