import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.denoiser import CryoEMDenoiser
from src.data_processing.cryoem_loader import HB5Dataset

def train_denoiser(dataset, epochs=3, batch_size=1):
    print("\n=== Initializing Training ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = CryoEMDenoiser().to(device)
    print("Model architecture:")
    print(model)
    
    # Test forward pass
    with torch.no_grad():
        test_input = torch.randn(1, 1, 64, 64, 64).to(device)
        test_output = model(test_input)
        print(f"\nTest input shape: {test_input.shape}")
        print(f"Test output shape: {test_output.shape}")
        assert test_output.shape == test_input.shape, "Input/output shape mismatch!"
    
    # Rest of training code remains the same...
    # Training setup
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
            cryoem = batch['cryoem'].unsqueeze(1).to(device)  # Add channel dim
            structure = batch['structure'].unsqueeze(1).to(device)
            
            # Add noise and denoise
            noise = torch.randn_like(cryoem) * 0.1
            noisy = cryoem + noise
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, structure)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")
    
    return model

def main():
    print("=== Protein Folding Training ===")
    try:
        # Load dataset
        dataset = HB5Dataset('data/raw/1hb5.cif')
        sample = dataset[0]
        print("\nDataset sample:")
        print(f"CryoEM shape: {sample['cryoem'].shape}")
        print(f"Structure shape: {sample['structure'].shape}")
        
        # Start training
        train_denoiser(dataset, epochs=3)
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()