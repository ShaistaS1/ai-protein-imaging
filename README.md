# 🧬 AI-Protein-Imaging
**AI-Augmented Hybrid Imaging for Ab Initio Protein Folding Without MSAs**

---

## 📌 Project Overview
This project aims to **predict 3D protein structures directly from sparse, noisy Cryo-EM and X-ray data** — without relying on **Multiple Sequence Alignments (MSAs)**, which are a core limitation in current models like AlphaFold.

- **Input:**  
  - Cryo-EM 2D class averages (low-resolution)  
  - Partial X-ray crystallography maps (incomplete data)  

- **Model:**  
  - 3D diffusion transformer (similar to AlphaFold3, conditioned on imaging data)  
  - Neural Cryo-EM denoiser (pre-trained to "hallucinate" missing density)  

- **Output:**  
  - Full atomic structures, even for orphan proteins with no evolutionary relatives.

---

## 👩‍💻 Author
**Shaista Aben e Azar**  
Final Year B.Sc Software Engineering  
University of Engineering and Technology, Taxila

---

## 🔬 Background
| Aspect                     | Vedula et al. (2025) - Prior Work | My Proposal - Next Frontier |
|-----------------------------|-----------------------------------|-----------------------------|
| **Core Problem**            | MSA failure in chimeric proteins  | Fundamental MSA dependence |
| **Primary Input**           | Evolutionary data (MSAs)          | Physical data (Cryo-EM, X-ray) |
| **Solution Approach**       | Fix MSA process                   | Eliminate MSA requirement |
| **Key Innovation**          | Sub-MSA merging                   | Diffusion model + imaging |
| **Scope of Application**    | Chimeric proteins with homologs    | Orphan proteins, synthetic proteins |
| **Principle**               | Evolutionary bioinformatics       | Computational physics + imaging |

---

## 🛠️ Data Pipeline
1. **Raw CIF file → 3D volume conversion**  
   - 3D scatter plot of atomic positions  
   - Orthogonal slices through volume  

2. **Synthetic Cryo-EM generation**  
   - Added realistic noise artifacts  
   - Side-by-side clean vs noisy slices  

3. **Input/Target Pairs**  
   - Model receives noisy input  
   - Learns to predict clean structure  

---

## ⚙️ Model Architecture
- Diffusion process: transforms random noise into protein structures  
- Conditioned on Cryo-EM/X-ray features  
- Transformer layers capture 3D atomic relationships  
- Cross-attention fuses modalities:
  - Cryo-EM → low-res global structure  
  - X-ray → high-res local structure  

---

## 📊 Example Visualizations

| Raw CIF Atoms | 3D Volume Conversion | Synthetic Cryo-EM |
|---------------|----------------------|------------------|
| <img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/ca454497-6695-4d22-b0c4-588dd9e4b1f8" /> | <img width="975" height="651" alt="image" src="https://github.com/user-attachments/assets/b6cd00f1-554c-4091-9200-080d15233b35" /> | <img width="975" height="412" alt="image" src="https://github.com/user-attachments/assets/d0ae174e-aaf9-47aa-8c3f-c76991e8bae5" />
 |

---

## 🚀 How to Run
Clone the repo and install dependencies:
```bash
git clone https://github.com/ShaistaS1/ai-protein-imaging.git
cd ai-protein-imaging
pip install -r requirements.txt
