import torch
import torch.nn as nn

class CryoEMDenoiser(nn.Module):
    def __init__(self, in_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder with same padding to maintain dimensions
        self.encoder.append(
            nn.Sequential(
                nn.Conv3d(in_channels, features[0], 3, padding=1),
                nn.BatchNorm3d(features[0]),
                nn.LeakyReLU()
            )
        )
        
        self.encoder.append(
            nn.Sequential(
                nn.Conv3d(features[0], features[1], 3, padding=1),
                nn.BatchNorm3d(features[1]),
                nn.LeakyReLU()
            )
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features[1], features[2], 3, padding=1),
            nn.BatchNorm3d(features[2]),
            nn.LeakyReLU()
        )
        
        # Decoder
        self.decoder.append(
            nn.Sequential(
                nn.Conv3d(features[2], features[1], 3, padding=1),
                nn.BatchNorm3d(features[1]),
                nn.LeakyReLU()
            )
        )
        
        self.decoder.append(
            nn.Sequential(
                nn.Conv3d(features[1], features[0], 3, padding=1),
                nn.BatchNorm3d(features[0]),
                nn.LeakyReLU()
            )
        )
        
        self.final_conv = nn.Conv3d(features[0], 1, 3, padding=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        
        # Bottleneck
        x = self.bottleneck(x2)
        
        # Decoder
        x = self.decoder[0](x)
        x = self.decoder[1](x)
        
        return torch.sigmoid(self.final_conv(x))