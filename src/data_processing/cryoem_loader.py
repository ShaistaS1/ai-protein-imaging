import numpy as np
import mrcfile
from scipy import ndimage
import torch
from torch.utils.data import Dataset
from Bio.PDB import MMCIFParser
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class HB5Dataset(Dataset):
    def __init__(self, cif_path, transform=None, grid_size=(64, 64, 64), resolution=3.0):
        """
        Args:
            cif_path (str): Path to 1hb5.cif
            transform (callable): Optional transform
            grid_size (tuple): Size of 3D volume
            resolution (float): Angstroms per voxel
        """
        self.cif_path = cif_path
        self.transform = transform
        self.grid_size = grid_size
        self.resolution = resolution
        
        # Load structure using Biopython
        parser = MMCIFParser()
        self.structure = parser.get_structure("1HB5", cif_path)
        
    def __len__(self):
        return 1  # Single structure dataset
    
    def __getitem__(self, idx):
        # Generate synthetic cryo-EM-like data
        volume = self._structure_to_volume()
        coords = self._get_atom_coords()
        
        if self.transform:
            volume = self.transform(volume)
            
        return {
            'cryoem': torch.tensor(volume, dtype=torch.float32),  # [64,64,64]
            'structure': torch.tensor(volume, dtype=torch.float32)  # Using same volume as target
        }
    
    def _structure_to_volume(self):
        """Convert atomic structure to 3D density grid"""
        volume = np.zeros(self.grid_size)
        
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        pos = atom.coord / self.resolution
                        ix, iy, iz = map(int, pos)
                        if all(0 <= p < s for p, s in zip((ix, iy, iz), self.grid_size)):
                            volume[ix, iy, iz] += 1
                            
        return ndimage.gaussian_filter(volume, sigma=1.5)
    
    def _get_atom_coords(self):
        """Extract all atom coordinates"""
        coords = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords.append(atom.coord)
        return np.array(coords)