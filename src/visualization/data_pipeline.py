import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import MMCIFParser
import torch  # Added this import

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.data_processing.cryoem_loader import HB5Dataset

def plot_atoms(cif_path):
    """Visualize atomic positions from CIF file"""
    parser = MMCIFParser()
    structure = parser.get_structure("1HB5", cif_path)
    
    # Extract coordinates
    coords = []
    for atom in structure.get_atoms():
        coords.append(atom.coord)
    coords = np.array(coords)
    
    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=10, alpha=0.5)
    ax.set_title("Atomic Positions from CIF File")
    plt.tight_layout()
    plt.show()

def plot_volume_slices(volume, title=""):
    """Show orthogonal slices through 3D volume"""
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    slices = [
        volume[volume.shape[0]//2, :, :],  # XY plane
        volume[:, volume.shape[1]//2, :],  # XZ plane
        volume[:, :, volume.shape[2]//2]   # YZ plane
    ]
    for ax, slc, plane in zip(axes, slices, ['XY', 'XZ', 'YZ']):
        ax.imshow(slc, cmap='viridis')
        ax.set_title(f'{plane} Plane')
    plt.suptitle(title)
    plt.show()

def interactive_3d_view(cif_path):
    """Interactive 3D view using nglview"""
    import nglview as nv
    parser = MMCIFParser()
    structure = parser.get_structure("1HB5", cif_path)
    view = nv.show_biopython(structure)
    view.add_representation('ball+stick', selection='all')
    return view

def showcase_pipeline(cif_path):
    print("=== Data Processing Pipeline ===")
    
    # 1. Show raw CIF structure
    print("\n1. Visualizing atomic structure from CIF file...")
    plot_atoms(cif_path)
    
    # 2. Show converted 3D volume
    print("\n2. Converting to 3D density volume...")
    dataset = HB5Dataset(cif_path)
    sample = dataset[0]
    plot_volume_slices(sample['cryoem'].numpy(), "Converted 3D Volume")
    
    # 3. Show synthetic cryo-EM generation
    print("\n3. Generating synthetic cryo-EM...")
    noisy = sample['cryoem'] + torch.randn_like(sample['cryoem']) * 0.1
    plot_volume_slices(noisy.numpy(), "Synthetic Cryo-EM with Noise")
    
    # 4. Show input/target pair
    print("\n4. Input/Target Pair:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(sample['cryoem'].numpy()[32], cmap='viridis')
    ax1.set_title("Input (XY slice)")
    ax2.imshow(sample['structure'].numpy()[32], cmap='viridis')
    ax2.set_title("Target (XY slice)")
    plt.show()
    
    # 5. Interactive 3D view (only works in Jupyter)
    print("\nNote: Interactive 3D viewer requires Jupyter Notebook")
    print("To view interactively, run this in a Jupyter cell:")
    print("from src.visualization.data_pipeline import interactive_3d_view")
    print("view = interactive_3d_view('data/raw/1hb5.cif')")
    print("view")

if __name__ == "__main__":
    showcase_pipeline('data/raw/1hb5.cif')