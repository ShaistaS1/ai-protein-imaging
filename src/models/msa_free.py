import torch
import torch.nn as nn

class MSAFreeEncoder(nn.Module):
    """Properly implemented windowed attention module"""
    def __init__(self, embed_dim=128, num_heads=8, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # Input projection
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=3, padding=1)
        
        # Transformer configuration
        self.attention = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,  # Critical fix for the warning
            activation='gelu'
        )
        
        # Position embeddings match window size
        self.position_embed = nn.Parameter(
            torch.randn(1, embed_dim, window_size, window_size, window_size)
        )

    def window_partition(self, x):
        """Split into non-overlapping windows"""
        B, C, D, H, W = x.shape
        x = x.view(B, C, 
                   D // self.window_size, self.window_size,
                   H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # [B, nD, nH, nW, ws, ws, ws, C]
        x = x.reshape(-1, self.window_size**3, self.embed_dim)
        return x

    def window_reverse(self, windows, D, H, W):
        """Reconstruct from windows"""
        B = int(windows.shape[0] / (D * H * W / self.window_size**3))
        x = windows.view(B, 
                         D // self.window_size,
                         H // self.window_size,
                         W // self.window_size,
                         self.window_size,
                         self.window_size,
                         self.window_size,
                         -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, -1, D, H, W)
        return x

    def forward(self, x):
        # Input: [B, D, H, W, 1]
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        
        # 1. Project to embedding space
        x = self.proj(x)  # [B, C, D, H, W]
        
        # 2. Window partition
        B, C, D, H, W = x.shape
        x_windows = self.window_partition(x)  # [nW*B, ws^3, C]
        
        # 3. Add position embeddings
        pos_embed = self.position_embed.view(1, -1, self.window_size**3).permute(0, 2, 1)
        x_windows = x_windows + pos_embed
        
        # 4. Apply attention
        x_windows = self.attention(x_windows)
        
        # 5. Reconstruct windows
        x = self.window_reverse(x_windows, D, H, W)
        
        return x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]

if __name__ == "__main__":
    # Test configuration
    encoder = MSAFreeEncoder(window_size=8)
    test_input = torch.randn(2, 32, 32, 32, 1)  # Must be divisible by window_size
    
    # Warmup run
    with torch.no_grad():
        _ = encoder(test_input)
    
    # Proper run
    output = encoder(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")