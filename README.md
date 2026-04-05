DeepPeptide: A Computational Molecular Evolution Engine for Exploring Functional Antimicrobial Peptide Space

Overview:
DeepPeptide is a hybrid deep learning framework for the design of antimicrobial peptides (AMPs). It combines a Conditional Variational Autoencoder (CVAE) with a potency-conditioned Long Short-Term Memory (LSTM) model to generate novel AMPs with predicted antimicrobial activity.

Requirements:
Before running the scripts, ensure the following dependencies are installed:
- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn
- Biopython (for sequence handling)

To install the dependencies, run:

pip install -r requirements.txt

Usage:
1. Clone this repository and install dependencies:

git clone https://github.com/gallactuz/DeepPeptide
cd DeepPeptide
pip install -r requirements.txt

2. To generate novel peptides conditioned on antimicrobial potency, run:

python3 deep_peptide_designer.py -vae bigdatabase.fasta -lstm micdatabase.fasta --length 15 --num 10 --target_potency 1.0 --alpha 0.9

Output:
The output files include:
- Generated Peptides: Sequences generated based on the chosen parameters.
- Evaluation Results: A summary of peptide properties, including net charge, hydrophobicity, and predicted antimicrobial activity.

License:
This project is licensed under the MIT License - see the LICENSE file for details.
