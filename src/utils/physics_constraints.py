import torch
import numpy as np
from Bio.PDB import PPBuilder
from collections import defaultdict

class PhysicsConstraints:
    def __init__(self):
        # Bond lengths in Angstroms (from molecular biology references)
        self.bond_lengths = {
            'C-N': 1.32,  # Peptide bond
            'C-C': 1.53,  # Standard carbon-carbon
            'N-CA': 1.47,
            'CA-C': 1.53,
            'C-O': 1.24
        }
        
        # Van der Waals radii (in Angstroms)
        self.vdw_radii = {
            'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8
        }
        
        # Load amino acid physical properties
        self.aa_properties = self._load_aa_data()
        
    def _load_aa_data(self):
        """Returns dictionary of amino acid properties"""
        return {
            'ALA': {'mass': 89.09, 'charge': 0, 'hydrophobicity': 1.8},
            'ARG': {'mass': 174.20, 'charge': +1, 'hydrophobicity': -4.5},
            'ASN': {'mass': 132.12, 'charge': 0, 'hydrophobicity': -3.5},
            'ASP': {'mass': 133.10, 'charge': -1, 'hydrophobicity': -3.5},
            'CYS': {'mass': 121.15, 'charge': 0, 'hydrophobicity': 2.5},
            'GLN': {'mass': 146.15, 'charge': 0, 'hydrophobicity': -3.5},
            'GLU': {'mass': 147.13, 'charge': -1, 'hydrophobicity': -3.5},
            'GLY': {'mass': 75.07, 'charge': 0, 'hydrophobicity': -0.4},
            'HIS': {'mass': 155.16, 'charge': 0, 'hydrophobicity': -3.2},
            'ILE': {'mass': 131.17, 'charge': 0, 'hydrophobicity': 4.5},
            'LEU': {'mass': 131.17, 'charge': 0, 'hydrophobicity': 3.8},
            'LYS': {'mass': 146.19, 'charge': +1, 'hydrophobicity': -3.9},
            'MET': {'mass': 149.21, 'charge': 0, 'hydrophobicity': 1.9},
            'PHE': {'mass': 165.19, 'charge': 0, 'hydrophobicity': 2.8},
            'PRO': {'mass': 115.13, 'charge': 0, 'hydrophobicity': -1.6},
            'SER': {'mass': 105.09, 'charge': 0, 'hydrophobicity': -0.8},
            'THR': {'mass': 119.12, 'charge': 0, 'hydrophobicity': -0.7},
            'TRP': {'mass': 204.23, 'charge': 0, 'hydrophobicity': -0.9},
            'TYR': {'mass': 181.19, 'charge': 0, 'hydrophobicity': -1.3},
            'VAL': {'mass': 117.15, 'charge': 0, 'hydrophobicity': 4.2}
        }
        
    def _get_atom_type(self, atom_name, residue_name):
        """Maps atom names to element types"""
        if atom_name.startswith('C'): return 'C'
        if atom_name.startswith('N'): return 'N'
        if atom_name.startswith('O'): return 'O'
        return 'C'  # Default
    
    def apply(self, coords, sequence):
        """
        Enforces physical constraints on atomic coordinates
        
        Args:
            coords: Tensor of shape [N_atoms, 3] 
            sequence: List of residue names (e.g., ['ALA', 'VAL'])
            
        Returns:
            Constrained coordinates
        """
        coords = coords.clone()  # Don't modify input
        N = coords.shape[0]
        
        with torch.no_grad():
            # 1. Bond length constraints
            residue_pointer = 0
            for res_idx, res_name in enumerate(sequence):
                # Get backbone atoms for this residue
                atoms = ['N', 'CA', 'C']
                if res_idx < len(sequence)-1:  # Not the last residue
                    atoms = ['N', 'CA', 'C']
                else:  # Last residue
                    atoms = ['N', 'CA', 'C', 'O']
                
                # Make sure we don't exceed coordinate tensor bounds
                if residue_pointer + len(atoms) > N:
                    break
                
                # Apply bond constraints
                for i in range(len(atoms)-1):
                    a1, a2 = atoms[i], atoms[i+1]
                    idx1, idx2 = residue_pointer+i, residue_pointer+i+1
                    
                    # Get ideal bond length
                    bond_type = f"{self._get_atom_type(a1,res_name)}-{self._get_atom_type(a2,res_name)}"
                    ideal_length = self.bond_lengths.get(bond_type, 1.5)
                    
                    # Adjust position
                    vec = coords[idx2] - coords[idx1]
                    current_length = torch.norm(vec)
                    if current_length > 0:
                        coords[idx2] = coords[idx1] + (vec/current_length)*ideal_length
                
                residue_pointer += len(atoms)
            
            # 2. Steric clashes avoidance
            for i in range(N):
                for j in range(i+1, min(i+5, N)):  # Skip adjacent atoms
                 dist = torch.norm(coords[i] - coords[j])
            atom_i_type = self._get_atom_type(f"Atom{i}", "XXX")
            atom_j_type = self._get_atom_type(f"Atom{j}", "XXX")
            min_dist = self.vdw_radii[atom_i_type] + self.vdw_radii[atom_j_type]
            
            if dist < min_dist:
                direction = coords[j] - coords[i]
                direction_norm = torch.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm  # Manual normalization
                    coords[j] += direction * (min_dist - dist) * 0.5
                    coords[i] -= direction * (min_dist - dist) * 0.5
    
            return coords

def generate_test_structure(length=10):
    """Creates a simple alpha-helix backbone"""
    # Helix parameters
    rise_per_residue = 1.5  # Å
    residues_per_turn = 3.6
    radius = 2.3  # Å
    
    coords = []
    sequence = ['ALA'] * length
    for i in range(length):
        angle = 2 * np.pi * i / residues_per_turn
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = i * rise_per_residue
        # Backbone atoms (N, CA, C)
        coords.extend([[x,y,z], [x,y,z+0.5], [x,y,z+1.0]])
        # Add O atom for last residue
        if i == length-1:
            coords.append([x,y,z+1.2])
    
    return torch.tensor(coords, dtype=torch.float32), sequence

if __name__ == "__main__":
    # Initialize constraints
    pc = PhysicsConstraints()
    
    # Test case 1: Random coil
    print("=== Testing Random Coil ===")
    random_coords = torch.randn(31, 3) * 5.0  # 10 residues (3 atoms each) + 1 O for last residue
    random_seq = ['ALA'] * 10
    constrained = pc.apply(random_coords, random_seq)
    print(f"Random -> Constrained distance change: {torch.norm(random_coords - constrained):.2f}Å")
    
    # Test case 2: Generated helix
    print("\n=== Testing Alpha Helix ===")
    helix_coords, helix_seq = generate_test_structure(10)
    perturbed = helix_coords + torch.randn_like(helix_coords) * 0.5
    reconstructed = pc.apply(perturbed, helix_seq)
    print(f"Reconstruction error: {torch.norm(helix_coords - reconstructed):.2f}Å")