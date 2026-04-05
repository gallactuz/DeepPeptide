"""
deep_peptide_designer.py — CVAE + LSTM orchestrator for AMP design.

Basic usage:
    python deep_peptide_designer.py \
        -vae bigdatabase.fasta \
        -lstm mic_database.fasta \
        --length 15 --num 10 \
        --target_potency 1.0      # generate sequences close to the most potent entries

Multiple small databases:
    python deep_peptide_designer.py \
        -vae bigdatabase.fasta \
        -lstm mic_database.fasta \
        -lstm gram_negative.fasta \
        --length 15 --num 10 \
        --target_potency 0.9
"""

import argparse
import logging
import sys
import csv
import numpy as np
from pathlib import Path

from cvae_model import CVAEModel, raw_properties, compute_pI, AA_DICT
from lstm_model import LSTMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('DeepPeptideDesigner')


class DeepPeptideDesigner:
    """
    Combines a CVAE (large database) with N potency-conditioned LSTMs (small databases).

    Assumptions about small databases:
        Sequences are SORTED from least potent to most potent
        (e.g. descending MIC → ascending potency).
        The normalised file position [0, 1] is used as the potency rank condition.

    Generation pipeline:
        1. CVAE   : z ~ N(0, I) → P_vae  (base distribution, max_length × 20).
        2. LSTMs  : seed from target potency range → P_lstm_i (target_length × 20).
        3. _combine(): weighted Product of Experts fusion.
        4. Physicochemical filters applied to candidate sequences.

    Args:
        vae_fasta   : Path to the large FASTA file for CVAE pre-training.
        lstm_fastas : List of paths to sorted small FASTA databases.
        alpha       : Weight of the CVAE distribution (0 = LSTM only, 1 = CVAE only).
        max_length  : Maximum peptide length shared by all models.
        latent_dim  : CVAE latent space dimensionality.
        seed        : Optional random seed for reproducibility.
    """

    def __init__(self, vae_fasta: str, lstm_fastas: list,
                 alpha: float = 0.5, max_length: int = 50,
                 latent_dim: int = 128, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Reproducibility seed set to: {seed}")

        self.alpha      = alpha
        self.max_length = max_length
        self.aa_dict    = AA_DICT
        self.aa_list    = list(AA_DICT.keys())
        self.num_aa     = len(AA_DICT)

        # Instantiate CVAE and one LSTM per small database
        self.cvae  = CVAEModel(vae_fasta, max_length=max_length,
                               latent_dim=latent_dim)
        self.lstms = [LSTMModel(f, max_length=max_length) for f in lstm_fastas]
        self._lstm_weights = None   # computed after training based on dataset sizes

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, vae_epochs=100, lstm_epochs=200, batch_size=32):
        """
        Load data and train all models (CVAE + all LSTMs).
        LSTM weights are set proportional to each database size (larger = higher weight).
        """
        logger.info("── Loading and training CVAE ──")
        X_vae, props_vae = self.cvae.load_data()
        self.cvae.train(X_vae, props_vae, epochs=vae_epochs, batch_size=batch_size)

        logger.info("── Loading and training LSTMs ──")
        sizes = []
        for i, lstm in enumerate(self.lstms):
            logger.info(f"LSTM {i+1}/{len(self.lstms)}: {lstm.small_fasta}")
            X, y, cond = lstm.load_data()
            lstm.train(X, y, cond, epochs=lstm_epochs, batch_size=batch_size)
            sizes.append(lstm.n_sequences)

        # Compute LSTM blend weights proportional to database size
        sizes = np.array(sizes, dtype=np.float32)
        self._lstm_weights = sizes / sizes.sum() if sizes.sum() > 0 else \
                             np.ones(len(sizes)) / len(sizes)

        for i, (lstm, w) in enumerate(zip(self.lstms, self._lstm_weights)):
            logger.info(f"  LSTM {i+1} ({Path(lstm.small_fasta).name}): "
                        f"{lstm.n_sequences} sequences → weight {w:.3f}")
        logger.info("All models trained successfully.")

    # ── Physicochemical properties ────────────────────────────────────────────

    @staticmethod
    def calculate_properties(sequence: str) -> dict:
        """
        Return a dict of physicochemical properties for a generated peptide.
        Keys: charge, pI, mass, hydrophobicity, length.
        """
        props = raw_properties(sequence)
        return {
            'charge':         float(props[0]),
            'pI':             float(props[1]),
            'mass':           float(props[2]),
            'hydrophobicity': float(props[3]),
            'length':         len(sequence),
        }

    # ── Distribution fusion ───────────────────────────────────────────────────

    def _combine(self, p_vae: np.ndarray,
                 p_lstms: list,
                 target_length: int) -> np.ndarray:
        """
        Fuse CVAE and LSTM distributions via a weighted Product of Experts:
            log P = alpha * log P_vae + (1 - alpha) * sum_i( w_i * log P_lstm_i )

        Args:
            p_vae         : CVAE output, shape (max_length, num_aa).
            p_lstms       : List of LSTM outputs, each shape (target_length, num_aa).
            target_length : Final sequence length to generate.

        Returns:
            Combined probability matrix, shape (target_length, num_aa).
        """
        p_vae_crop = p_vae[:target_length]  # trim CVAE output to target length

        if p_lstms and self._lstm_weights is not None:
            weights    = self._lstm_weights[:len(p_lstms)]
            weights    = weights / weights.sum()             # re-normalise weights
            log_p_lstm = np.zeros_like(p_lstms[0])
            for w, p_l in zip(weights, p_lstms):
                ml = min(p_vae_crop.shape[0], p_l.shape[0])
                log_p_lstm[:ml] += w * np.log(np.clip(p_l[:ml], 1e-8, 1.0))

            ml           = min(p_vae_crop.shape[0], log_p_lstm.shape[0])
            log_p_vae    = np.log(np.clip(p_vae_crop[:ml], 1e-8, 1.0))
            combined_log = self.alpha * log_p_vae + (1 - self.alpha) * log_p_lstm[:ml]
            combined     = np.exp(combined_log)
        else:
            # Fall back to CVAE only when no LSTMs are available
            combined = p_vae_crop

        # Row-normalise to ensure valid probability distributions
        sums     = combined.sum(axis=-1, keepdims=True)
        combined = combined / np.where(sums == 0, 1, sums)
        return combined

    # ── Sequence sampling ─────────────────────────────────────────────────────

    @staticmethod
    def _sample_sequence(probs: np.ndarray, aa_list: list,
                         temperature: float = 1.0) -> str:
        """
        Sample one amino acid per position from the combined probability matrix.

        Args:
            probs       : (target_length, num_aa) probability array.
            aa_list     : Ordered list of amino acid characters.
            temperature : Sampling temperature (same semantics as in generate()).

        Returns:
            Sampled peptide string.
        """
        sequence = []
        for dist in probs:
            if temperature != 1.0:
                logits  = np.log(np.clip(dist, 1e-8, 1.0)) / temperature
                logits -= logits.max()   # numerical stability
                dist    = np.exp(logits)
                dist   /= dist.sum()
            idx = np.random.choice(len(aa_list), p=dist)
            sequence.append(aa_list[idx])
        return ''.join(sequence)

    # ── Peptide design ────────────────────────────────────────────────────────

    def design_peptide(self,
                       target_length: int = 15,
                       num_peptides: int = 5,
                       min_charge: float = 1.0,
                       min_hydrophobicity: float = 0.0,
                       temperature: float = 1.0,
                       target_properties: dict = None,
                       target_potency: float = 1.0,
                       max_attempts: int = 50) -> list:
        """
        Generate candidate antimicrobial peptides.

        Args:
            target_length      : Desired peptide length (number of residues).
            num_peptides       : Number of peptide candidates to return.
            min_charge         : Minimum net charge filter (cationic AMPs requirement).
            min_hydrophobicity : Minimum cumulative hydrophobicity filter.
            temperature        : Sampling temperature for both CVAE and LSTM outputs.
            target_properties  : Optional dict with CVAE conditioning values
                                 (keys: charge, pI, mass, hydrophobicity).
            target_potency     : [0.0, 1.0] — target position in the potency ranking.
                                 1.0 → generate near the most potent database entries.
                                 0.5 → median potency range.
                                 0.0 → least potent (rarely useful).
            max_attempts       : Maximum sampling attempts before accepting the best
                                 candidate regardless of filter criteria.

        Returns:
            List of dicts with keys: sequence, charge, pI, mass, hydrophobicity, length.
        """
        # ── Build CVAE conditioning vector ────────────────────────────────────
        raw_cond = np.array([
            (target_properties or {}).get('charge',         2.0),
            (target_properties or {}).get('pI',             8.0),
            (target_properties or {}).get('mass',        1500.0),
            (target_properties or {}).get('hydrophobicity', 5.0),
        ], dtype=np.float32)
        norm_cond_cvae = self.cvae.normalize_condition(raw_cond)

        logger.info(f"Target potency for LSTMs: {target_potency:.2f} "
                    f"({'most potent' if target_potency >= 0.8 else 'moderate' if target_potency >= 0.4 else 'least potent'})")

        # ── Build LSTM conditioning vectors (one per database) ────────────────
        lstm_conds = []
        for lstm in self.lstms:
            seed_props = raw_properties(lstm.seed_by_potency(target_potency))
            lstm_conds.append(
                lstm.normalize_condition(seed_props, target_potency=target_potency)
            )

        peptides = []
        for idx in range(num_peptides):
            best       = None
            best_props = None

            for _ in range(max_attempts):
                # Sample a latent vector from the CVAE prior
                z     = np.random.normal(
                    size=(1, self.cvae.latent_dim)).astype(np.float32)
                p_vae = self.cvae.generate(norm_cond_cvae,
                                           latent_sample=z,
                                           temperature=temperature)

                # Generate LSTM distributions for each database
                p_lstms = []
                for lstm, cond in zip(self.lstms, lstm_conds):
                    p_l = lstm.generate(
                        target_length  = target_length,
                        condition      = cond,
                        temperature    = temperature,
                        target_potency = target_potency   # seed from the correct potency window
                    )
                    p_lstms.append(p_l)

                # Fuse distributions and sample a candidate sequence
                combined   = self._combine(p_vae, p_lstms, target_length)
                seq        = self._sample_sequence(combined, self.aa_list, temperature)
                props      = self.calculate_properties(seq)
                best       = seq
                best_props = props

                # Accept if physicochemical criteria are satisfied
                if (props['charge'] >= min_charge and
                        props['hydrophobicity'] >= min_hydrophobicity):
                    break
            else:
                # max_attempts reached — log warning and keep best candidate
                logger.warning(
                    f"Peptide #{idx+1}: criteria not met after "
                    f"{max_attempts} attempts — best candidate accepted.")

            entry = {'sequence': best, **best_props}
            peptides.append(entry)
            logger.info(
                f"AMP #{idx+1}: {best} | "
                f"charge={best_props['charge']:.2f} "
                f"pI={best_props['pI']:.2f} "
                f"mass={best_props['mass']:.1f} "
                f"hydro={best_props['hydrophobicity']:.2f}")

        return peptides


# ── Output helpers ────────────────────────────────────────────────────────────

def save_results(peptides: list, output_path: str):
    """
    Save designed peptides to both CSV and FASTA formats.

    CSV columns : sequence, charge, pI, mass, hydrophobicity, length.
    FASTA header: >AMP_<n> charge=... pI=... mass=... hydro=...
    """
    p = Path(output_path)

    # Write CSV file
    with open(p, 'w', newline='') as f:
        fields = ['sequence', 'charge', 'pI', 'mass', 'hydrophobicity', 'length']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(peptides)
    logger.info(f"CSV saved: {p}")

    # Write FASTA file alongside the CSV
    fasta_path = p.with_suffix('.fasta')
    with open(fasta_path, 'w') as f:
        for i, pep in enumerate(peptides, 1):
            f.write(f">AMP_{i} charge={pep['charge']:.2f} pI={pep['pI']:.2f} "
                    f"mass={pep['mass']:.1f} hydro={pep['hydrophobicity']:.2f}\n")
            f.write(pep['sequence'] + '\n')
    logger.info(f"FASTA saved: {fasta_path}")


# ── Command-line interface ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Design AMPs with a potency-conditioned CVAE + LSTM pipeline.')

    # ── Required inputs ───────────────────────────────────────────────────────
    parser.add_argument('-vae',  required=True,
                        help='Large FASTA database for the CVAE.')
    parser.add_argument('-lstm', action='append', required=True,
                        help='Small FASTA database(s) SORTED by ascending potency '
                             '(can be specified multiple times).')

    # ── Fusion and generation parameters ─────────────────────────────────────
    parser.add_argument('--alpha',           type=float, default=0.5,
                        help='CVAE weight in the fusion (0 = LSTM only, 1 = CVAE only).')
    parser.add_argument('--length',          type=int,   default=15,
                        help='Target peptide length (number of residues).')
    parser.add_argument('--num',             type=int,   default=5,
                        help='Number of peptide candidates to generate.')
    parser.add_argument('--temperature',     type=float, default=1.0,
                        help='Sampling temperature (< 1 = more deterministic, > 1 = more diverse).')
    parser.add_argument('--max_length',      type=int,   default=50,
                        help='Maximum sequence length for model encoding.')

    # ── Physicochemical filters ───────────────────────────────────────────────
    parser.add_argument('--min_charge',         type=float, default=1.0,
                        help='Minimum net charge required (cationicity filter).')
    parser.add_argument('--min_hydrophobicity', type=float, default=0.0,
                        help='Minimum cumulative hydrophobicity required.')
    parser.add_argument('--max_attempts',       type=int,   default=50,
                        help='Max sampling attempts before accepting best candidate.')

    # ── Potency target ────────────────────────────────────────────────────────
    parser.add_argument('--target_potency',  type=float, default=1.0,
                        help='Target potency rank [0.0–1.0]. '
                             '1.0 = most potent (default). '
                             'Reflects position in the sorted FASTA file.')

    # ── Physicochemical targets for CVAE conditioning ─────────────────────────
    parser.add_argument('--target_charge',         type=float, default=None,
                        help='Target net charge for CVAE conditioning.')
    parser.add_argument('--target_pI',             type=float, default=None,
                        help='Target isoelectric point for CVAE conditioning.')
    parser.add_argument('--target_mass',           type=float, default=None,
                        help='Target molecular mass (Da) for CVAE conditioning.')
    parser.add_argument('--target_hydrophobicity', type=float, default=None,
                        help='Target hydrophobicity for CVAE conditioning.')

    # ── Training hyperparameters ──────────────────────────────────────────────
    parser.add_argument('--vae_epochs',  type=int, default=100,
                        help='Training epochs for the CVAE.')
    parser.add_argument('--lstm_epochs', type=int, default=200,
                        help='Training epochs for each LSTM.')
    parser.add_argument('--batch_size',  type=int, default=32,
                        help='Batch size for all model training.')
    parser.add_argument('--latent_dim',  type=int, default=128,
                        help='CVAE latent space dimensionality.')

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',   type=int, default=None,
                        help='Random seed for reproducibility.')
    parser.add_argument('--output', type=str, default='designed_amps.csv',
                        help='Output CSV path (a FASTA file is also written).')

    args = parser.parse_args()

    # Build optional CVAE target-properties dict
    target_props = None
    if any(v is not None for v in [args.target_charge, args.target_pI,
                                    args.target_mass, args.target_hydrophobicity]):
        target_props = {
            'charge':         args.target_charge         or 2.0,
            'pI':             args.target_pI             or 8.0,
            'mass':           args.target_mass           or 1500.0,
            'hydrophobicity': args.target_hydrophobicity or 5.0,
        }
        logger.info(f"CVAE target properties: {target_props}")

    # Instantiate and train the designer
    designer = DeepPeptideDesigner(
        vae_fasta  = args.vae,
        lstm_fastas= args.lstm,
        alpha      = args.alpha,
        max_length = args.max_length,
        latent_dim = args.latent_dim,
        seed       = args.seed
    )
    designer.initialize(
        vae_epochs  = args.vae_epochs,
        lstm_epochs = args.lstm_epochs,
        batch_size  = args.batch_size
    )

    # Generate peptide candidates
    peptides = designer.design_peptide(
        target_length     = args.length,
        num_peptides      = args.num,
        min_charge        = args.min_charge,
        min_hydrophobicity= args.min_hydrophobicity,
        temperature       = args.temperature,
        target_properties = target_props,
        target_potency    = args.target_potency,
        max_attempts      = args.max_attempts
    )

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"{'#':>3}  {'Sequence':<25} {'Charge':>7} {'pI':>6} "
          f"{'Mass':>8} {'Hydro':>7}")
    print("─" * 65)
    for i, p in enumerate(peptides, 1):
        print(f"{i:>3}  {p['sequence']:<25} {p['charge']:>7.2f} "
              f"{p['pI']:>6.2f} {p['mass']:>8.1f} {p['hydrophobicity']:>7.2f}")
    print("═" * 65)

    save_results(peptides, args.output)


if __name__ == '__main__':
    main()
