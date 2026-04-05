"""
lstm_model.py — Conditional autoregressive LSTM for AMP refinement.

Conditioning scheme (v3):
  - [charge, pI, mass, hydrophobicity, rank]
  - rank = normalised position in the file (0.0 = least potent, 1.0 = most potent)
  - Works with any sorted database: ascending MIC, IC50, activity score, etc.
  - During generation: target_potency=1.0 steers sampling towards the top of the file.
"""

import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Masking, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from Bio import SeqIO
from cvae_model import AA_DICT, raw_properties

logger = logging.getLogger(__name__)


class LSTMModel:
    """
    Conditional autoregressive LSTM for AMP sequence refinement.

    The small database is assumed to be SORTED from least potent to most potent
    (e.g. descending MIC → ascending potency). The normalised file position [0.0, 1.0]
    is used as the 5th conditioning dimension (potency rank).

    During generation, target_potency=1.0 produces sequences close to the file top.

    Args:
        small_fasta  : Path to the (sorted) small FASTA database.
        max_length   : Maximum peptide length for one-hot encoding.
        hidden_units : Number of units in the first LSTM layer.
    """

    def __init__(self, small_fasta: str, max_length: int = 50,
                 hidden_units: int = 256):
        self.small_fasta  = small_fasta
        self.max_length   = max_length
        self.hidden_units = hidden_units
        self.aa_dict      = AA_DICT
        self.num_aa       = len(AA_DICT)
        self.model        = None
        self.cond_dim     = 5          # [charge, pI, mass, hydrophobicity, rank]
        self._sequences   = None       # stores cleaned sequences after load_data()
        self.n_sequences  = 0          # exposed for external LSTM weighting

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _clean(self, seq: str) -> str:
        """Remove non-standard residues and convert to uppercase."""
        return ''.join(aa for aa in seq.upper() if aa in self.aa_dict)

    def _one_hot(self, seq: str) -> np.ndarray:
        """
        Encode a peptide sequence as a (max_length, num_aa) one-hot matrix.
        Positions beyond the sequence length are left as zero (padding).
        """
        enc = np.zeros((self.max_length, self.num_aa), dtype=np.float32)
        for i, aa in enumerate(seq[:self.max_length]):
            if aa in self.aa_dict:
                enc[i, self.aa_dict[aa]] = 1.0
        return enc

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self):
        """
        Load sequences and prepare autoregressive training pairs (X, y, cond).

        Condition vector per sequence:
            cond = [charge_norm, pI_norm, mass_norm, hydro_norm, rank_norm]
        where rank_norm = index / (N - 1), ranging from 0.0 (start of file)
        to 1.0 (end of file), reflecting ascending potency order.

        Returns:
            X    : np.ndarray (N, T-1, num_aa)  — input tokens (positions 0..L-2).
            y    : np.ndarray (N, T-1, num_aa)  — target tokens (positions 1..L-1).
            cond : np.ndarray (N, T-1, cond_dim) — condition tiled along time axis.
        """
        records = list(SeqIO.parse(self.small_fasta, 'fasta'))
        seqs    = [self._clean(str(r.seq)) for r in records]
        seqs    = [s for s in seqs if len(s) >= 4]   # discard very short peptides

        if len(seqs) < 2:
            raise ValueError(
                f"'{self.small_fasta}' requires at least 2 valid sequences.")

        self._sequences  = seqs
        self.n_sequences = len(seqs)
        n = len(seqs)

        # Ordinal rank: 0.0 = least potent (first entry), 1.0 = most potent (last)
        ranks = np.linspace(0.0, 1.0, n, dtype=np.float32)  # shape (N,)

        logger.info(
            f"LSTM ({self.small_fasta}): {n} sequences loaded | "
            f"rank 0.0 → '{seqs[0][:8]}...' (least potent) | "
            f"rank 1.0 → '{seqs[-1][:8]}...' (most potent)")

        # Compute and locally normalise physicochemical properties
        props_raw  = np.array([raw_properties(s) for s in seqs], dtype=np.float32)
        props_norm = self._local_normalize(props_raw)  # shape (N, 4)

        # Append potency rank as the 5th condition dimension
        cond_full = np.column_stack([props_norm, ranks])  # shape (N, 5)

        # Build autoregressive pairs
        X_list, y_list, cond_list = [], [], []
        for seq, cond in zip(seqs, cond_full):
            oh = self._one_hot(seq)                            # (max_length, 20)
            X_list.append(oh[:-1])                            # input:  tokens 0…L-2
            y_list.append(oh[1:])                             # target: tokens 1…L-1
            cond_list.append(
                np.tile(cond, (self.max_length - 1, 1)))      # broadcast over time

        X    = np.array(X_list,    dtype=np.float32)    # (N, T-1, 20)
        y    = np.array(y_list,    dtype=np.float32)    # (N, T-1, 20)
        cond = np.array(cond_list, dtype=np.float32)    # (N, T-1, 5)
        return X, y, cond

    # ── Local normalisation ───────────────────────────────────────────────────

    def _local_normalize(self, props: np.ndarray) -> np.ndarray:
        """
        Min-max normalise the four physicochemical properties (NOT the rank).
        Stores the per-feature min/max for use in normalize_condition().
        Handles constant features by widening the range by ±1.
        """
        mins = props.min(axis=0)
        maxs = props.max(axis=0)
        eq   = maxs == mins
        maxs[eq] += 1.0
        mins[eq] -= 1.0
        self._local_mins = mins
        self._local_maxs = maxs
        return (props - mins) / (maxs - mins)

    def normalize_condition(self, props_raw: np.ndarray,
                             target_potency: float = 1.0) -> np.ndarray:
        """
        Normalise physicochemical properties and append the target potency rank.

        Args:
            props_raw      : array of shape (4,) — [charge, pI, mass, hydrophobicity].
            target_potency : float in [0.0, 1.0].
                             0.0 = generate sequences close to the file start (least potent).
                             1.0 = generate sequences close to the file end (most potent).

        Returns:
            Normalised condition vector of shape (5,).
        """
        props_4n = ((props_raw - self._local_mins) /
                    (self._local_maxs - self._local_mins)).astype(np.float32)
        rank = np.clip(target_potency, 0.0, 1.0)   # clamp to valid range
        return np.append(props_4n, rank).astype(np.float32)

    # ── Potency-aware seed selection ──────────────────────────────────────────

    def seed_by_potency(self, target_potency: float = 1.0,
                        window: float = 0.2) -> str:
        """
        Return a random sequence sampled from the desired potency window.

        Example: target_potency=1.0, window=0.2
            → sample uniformly from the top 20% of the sorted file.
        Example: target_potency=0.5, window=0.2
            → sample from the 40%–60% range.

        Args:
            target_potency : Centre of the potency window [0.0, 1.0].
            window         : Width of the sampling window (fraction of dataset).

        Returns:
            A cleaned peptide string from the selected potency range.
        """
        if self._sequences is None:
            raise RuntimeError("Call load_data() before seed_by_potency().")
        n  = len(self._sequences)
        lo = max(0, int((target_potency - window / 2) * n))
        hi = min(n, int((target_potency + window / 2) * n) + 1)
        if lo >= hi:
            lo, hi = max(0, hi - 1), hi
        idx = np.random.randint(lo, hi)
        return self._sequences[idx]

    def random_seed(self) -> str:
        """Return a uniformly random sequence from the loaded dataset."""
        if self._sequences is None:
            raise RuntimeError("Call load_data() before random_seed().")
        return self._sequences[np.random.randint(len(self._sequences))]

    def last_sequence(self) -> str:
        """Return the last (most potent) sequence in the sorted dataset."""
        if self._sequences is None:
            raise RuntimeError("Call load_data() before last_sequence().")
        return self._sequences[-1]

    # ── Model architecture ────────────────────────────────────────────────────

    def build_model(self):
        """
        Construct the two-layer conditional LSTM.

        Input:
            lstm_seq  : (batch, T, num_aa)    — one-hot sequence tokens.
            lstm_cond : (batch, T, cond_dim)  — condition broadcast over time.

        Output:
            (batch, T, num_aa) — softmax distribution over amino acids at each step.
        """
        inp_seq  = Input(shape=(None, self.num_aa),   name='lstm_seq')
        inp_cond = Input(shape=(None, self.cond_dim), name='lstm_cond')  # (T, 5)

        # Concatenate amino-acid token with condition at each time step
        x   = Concatenate(axis=-1)([inp_seq, inp_cond])
        x   = Masking(mask_value=0.0)(x)                              # ignore padding
        x   = LSTM(self.hidden_units, return_sequences=True, name='lstm1')(x)
        x   = Dropout(0.3)(x)
        x   = LSTM(self.hidden_units // 2, return_sequences=True, name='lstm2')(x)
        x   = Dropout(0.2)(x)
        out = Dense(self.num_aa, activation='softmax', name='output')(x)

        self.model = Model([inp_seq, inp_cond], out, name='lstm_rank_conditioned')
        self.model.compile(
            optimizer=Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"LSTM model built (cond_dim={self.cond_dim}: "
                    f"charge + pI + mass + hydrophobicity + rank).")
        self.model.summary(print_fn=logger.info)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X, y, cond, epochs=200, batch_size=16):
        """
        Train the conditional LSTM on autoregressive sequence pairs.

        Callbacks:
            EarlyStopping     — monitors val_loss; stops after 40 stagnant epochs.
            ReduceLROnPlateau — halves LR after 8 epochs without improvement.
        Validation split is 15% when >= 20 sequences are available, otherwise 0.
        """
        if self.model is None:
            self.build_model()

        val_split = 0.15 if len(X) >= 20 else 0.0
        self.model.fit(
            [X, cond], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=40,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=8, min_lr=1e-6, verbose=1)
            ],
            verbose=1
        )

    # ── Autoregressive generation ─────────────────────────────────────────────

    def generate(self, target_length: int,
                 condition: np.ndarray,
                 temperature: float = 1.0,
                 target_potency: float = 1.0) -> np.ndarray:
        """
        Autoregressively generate a per-position AA probability distribution.

        The seed sequence is drawn from the potency window matching target_potency,
        ensuring the initial context is already consistent with the desired activity.

        Args:
            target_length  : Number of amino acids to generate.
            condition      : Normalised condition vector of shape (5,) (includes rank).
            temperature    : Sampling temperature.
                             < 1 → sharper / more deterministic.
                             > 1 → softer / more diverse.
            target_potency : Position [0.0, 1.0] in the sorted file to seed from.

        Returns:
            np.ndarray of shape (target_length, num_aa) — per-position AA probabilities.
        """
        if self.model is None:
            raise RuntimeError("Call train() before generate().")

        aa_list     = list(self.aa_dict.keys())
        seed        = self.seed_by_potency(target_potency)   # potency-consistent seed
        current_seq = list(seed[:self.max_length])
        all_probs   = []

        for _ in range(target_length):
            inp_oh   = self._one_hot(''.join(current_seq))[np.newaxis]        # (1, T, 20)
            inp_cond = np.tile(condition, (1, inp_oh.shape[1], 1)).astype(np.float32)

            # Predict next-token distribution from the last time step
            probs = self.model.predict([inp_oh, inp_cond], verbose=0)[0][-1]  # (20,)

            # Apply temperature rescaling when requested
            if temperature != 1.0:
                logits  = np.log(np.clip(probs, 1e-8, 1.0)) / temperature
                logits -= logits.max()    # numerical stability
                probs   = np.exp(logits)
                probs  /= probs.sum()

            all_probs.append(probs.copy())
            # Sample the next amino acid and append to the growing sequence
            next_idx = np.random.choice(self.num_aa, p=probs)
            current_seq.append(aa_list[next_idx])
            # Keep context window within max_length
            if len(current_seq) > self.max_length:
                current_seq = current_seq[-self.max_length:]

        return np.array(all_probs, dtype=np.float32)   # (target_length, 20)
