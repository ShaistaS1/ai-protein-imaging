import os
import requests
from torch.utils.data import Dataset
from Bio.PDB import PDBParser

class ChimericDataset(Dataset):
    def __init__(self, root_dir="data/chimeric"):
        self.root_dir = root_dir
        self.scaffolds = ['2IYDB', '1PKW']
        self._ensure_data_exists()  # Check and download missing files
        self.peptides = self._load_peptides(os.path.join(root_dir, "peptides"))
        self.pairs = self._generate_pairs()
        
    def _ensure_data_exists(self):
        """Ensure required directories and files exist"""
        os.makedirs(os.path.join(self.root_dir, "peptides"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "scaffolds"), exist_ok=True)
        
        # Download scaffold files if missing
        scaffolds = {
            '2IYDB': '2IYD',
            '1PKW': '1PKW'
        }
        
        # Example peptides to download
        peptides = ['1ABC', '1L2Y', '2NOV', '1FKG', '1PJE']
        
        # Download missing scaffold files
        for local_name, pdb_id in scaffolds.items():
            path = os.path.join(self.root_dir, "scaffolds", f"{local_name}.pdb")
            if not os.path.exists(path):
                self._download_pdb(pdb_id, path)
        
        # Download missing peptide files
        for pdb_id in peptides:
            path = os.path.join(self.root_dir, "peptides", f"{pdb_id}.pdb")
            if not os.path.exists(path):
                self._download_pdb(pdb_id, path)
    
    def _download_pdb(self, pdb_id, save_path):
        """Download a PDB file from RCSB"""
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {pdb_id} to {save_path}")
        except Exception as e:
            print(f"Failed to download {pdb_id}: {str(e)}")
        
    def _load_peptides(self, path):
        """Load peptide filenames from directory"""
        try:
            return [f.split('.')[0] for f in os.listdir(path) if f.endswith('.pdb')]
        except FileNotFoundError:
            print(f"Warning: Peptide directory not found at {path}")
            return []
        
    def _generate_pairs(self):
        return [(s, p) for s in self.scaffolds for p in self.peptides]
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        scaffold, peptide = self.pairs[idx]
        try:
            return {
                'sequence': f"{scaffold}-GS-{peptide}",
                'scaffold': self._load_structure(scaffold, "scaffolds"),
                'peptide': self._load_structure(peptide, "peptides")
            }
        except Exception as e:
            print(f"Error loading pair ({scaffold}, {peptide}): {str(e)}")
            return None
        
    def _load_structure(self, pdb_id, subdir):
        """Load PDB structure with proper path handling"""
        path = os.path.join(self.root_dir, subdir, f"{pdb_id}.pdb")
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDB file not found: {path}")
            
        parser = PDBParser(QUIET=True)  # Suppress warnings
        try:
            return parser.get_structure(pdb_id, path)
        except Exception as e:
            print(f"Error parsing {path}: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        dataset = ChimericDataset()
        if len(dataset) == 0:
            print("No valid protein pairs found. Check your data directory.")
        else:
            sample = dataset[0]
            if sample is not None:
                print(f"First sample: {sample['sequence']}")
                print(f"Scaffold atoms: {len(list(sample['scaffold'].get_atoms()))}")
                print(f"Peptide atoms: {len(list(sample['peptide'].get_atoms()))}")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        print("Please ensure:")
        print("1. You have internet connection for automatic downloads")
        print("2. The directory structure exists: data/chimeric/{peptides,scaffolds}")