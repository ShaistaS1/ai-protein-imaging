import matplotlib.pyplot as plt
import numpy as np
import os
from Bio.PDB import *
from Bio.PDB import PDBList

def create_sample_files():
    """Generate sample PDB files if none exist"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/true.pdb"):
        pdbl = PDBList()
        samples = {
            "1CRN": "true.pdb",
            "1L2Y": "pred.pdb",
            "1ZDD": "af.pdb"
        }
        for pdb_id, filename in samples.items():
            pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir="temp")
            os.rename(f"temp/pdb{pdb_id.lower()}.ent", f"data/{filename}")
        os.rmdir("temp")

def render_structure(structure):
    """Convert structure to density map"""
    # Simple example - replace with actual rendering
    return np.random.rand(64,64,64)

def compare_predictions(true_pdb, pred_pdb, af_pdb):
    fig = plt.figure(figsize=(15,5))
    
    volumes = [
        (render_structure(true_pdb), 'True Structure'),
        (render_structure(pred_pdb), 'Our Prediction'), 
        (render_structure(af_pdb), 'AlphaFold')
    ]
    
    for i, (vol, title) in enumerate(volumes):
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(vol[32], cmap='viridis')
        ax.set_title(title)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison.png")
    plt.close()
    return fig

if __name__ == "__main__":
    create_sample_files()
    
    parser = PDBParser()
    try:
        true = parser.get_structure("true", "data/true.pdb")
        pred = parser.get_structure("pred", "data/pred.pdb")
        af = parser.get_structure("af", "data/af.pdb")
        
        compare_predictions(true, pred, af)
        print("Successfully generated comparison.png in results/")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Using random data for visualization")
        # Fallback with random data
        compare_predictions(None, None, None)