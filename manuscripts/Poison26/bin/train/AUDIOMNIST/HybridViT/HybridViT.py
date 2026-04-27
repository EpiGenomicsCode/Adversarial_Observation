import torch
import torch.nn as nn

class HybridViT(nn.Module):
    """
    A Convolutional-Transformer Hybrid Architecture.
    Dynamically sizes its positional embeddings and patch count based on the input batch,
    making it easily adaptable to different image or spectrogram resolutions.
    """
    def __init__(self, one_batch=None, num_classes=10, embed_dim=64, num_heads=4, depth=2):
        super(HybridViT, self).__init__()
        
        # -------------------------
        # Dynamic Input Handling
        # -------------------------
        if one_batch is not None:
            _, in_channels, H, W = one_batch.shape
            self.input_channels = in_channels
            self.input_size = (in_channels, H, W)
        else:
            self.input_channels = 1
            self.input_size = (1, 28, 28) # Default to MNIST size

        self.embed_dim = embed_dim

        # -------------------------
        # 1. CNN Stem (Local Feature Extraction)
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # -------------------------
        # 2. Dynamic Patch Calculation
        # -------------------------
        self.num_patches = self._get_num_patches(one_batch)
        
        # Learnable positional embedding to retain spatial awareness
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
        
        # -------------------------
        # 3. Transformer Encoder (Global Context)
        # -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            dim_feedforward=self.embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # -------------------------
        # 4. Classifier Head
        # -------------------------
        self.norm = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    # -------------------------
    # Compute Patch Size Dynamically
    # -------------------------
    def _get_num_patches(self, one_batch):
        # Explicitly set eval() to prevent tracking BatchNorm stats with the dummy batch
        was_training = self.training
        self.eval()

        with torch.no_grad():
            if one_batch is None:
                dummy_input = torch.zeros(1, *self.input_size)
            else:
                _, C, H, W = one_batch.shape
                dummy_input = torch.zeros(1, C, H, W)

            # Pass through the stem
            x = self.stem(dummy_input)
            
            # Extract resulting spatial dimensions
            _, _, h_out, w_out = x.shape
            num_patches = h_out * w_out

        if was_training:
            self.train()

        return num_patches

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x):
        # 1. Pass through CNN stem
        x = self.stem(x) 
        
        # 2. Flatten spatial dimensions to create a sequence
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: (Batch, num_patches, embed_dim)
        
        # 3. Add positional embeddings
        x = x + self.pos_embed
        
        # 4. Pass through Transformer
        x = self.transformer(x)
        
        # 5. Global Average Pooling over the sequence length
        x = x.mean(dim=1)  # Shape: (Batch, embed_dim)
        
        # 6. Classify
        x = self.norm(x)
        x = self.fc(x)
        
        return x