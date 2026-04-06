DeepPeptide: A Computational Molecular Evolution Engine for Exploring Functional Antimicrobial Peptide Space

Overview
DeepPeptide is a hybrid deep learning framework for the design of antimicrobial peptides (AMPs). It combines a Conditional Variational Autoencoder (CVAE) with a potency-conditioned Long Short-Term Memory (LSTM) model to generate novel AMPs with predicted antimicrobial activity.

---

Requirements
Before running the scripts, make sure the following dependencies are installed:
- Python 3.9 or higher
- TensorFlow 2.x
- NumPy
- scikit-learn
- Biopython (for sequence handling)

To install all dependencies, run:
pip install -r requirements.txt

Usage

1. Clone and Install
git clone https://github.com/gallactuz/DeepPeptide
cd DeepPeptide
pip install -r requirements.txt

2. Generate Novel Peptides
python3 deep_peptide_designer.py \
  -vae bigdatabase.fasta \
  -lstm micdatabase.fasta \
  --length 15 \
  --num 10 \
  --target_potency 1.0 \
  --alpha 0.9

Output
The program generates the following files:
- designed_amps.csv — Table containing sequences and their physicochemical properties (charge, pI, mass, hydrophobicity)
- designed_amps.fasta — Generated peptide sequences in FASTA format

Main Parameters
--alpha: Weight of the CVAE in the hybrid generation (0 = LSTM only, 1 = CVAE only)
--target_potency: Desired potency level (1.0 = highest potency)
--length: Length of the peptides to generate
--num: Number of peptides to generate
--temperature: Controls creativity/randomness of generation (0.8–1.0)
--vae_epochs / --lstm_epochs: Number of training epochs (reduce significantly when using CPU)

---
To validate and rank the generated peptides, run:
python3 validate_amps.py
---
License
This project is licensed under the MIT License — see the LICENSE file for details.
