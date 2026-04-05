"""
cvae_model.py — Conditional Variational Autoencoder (CVAE) for AMP design.
Compatible with Keras 3.
"""

import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Flatten, Reshape,
    Conv1D, MaxPooling1D, UpSampling1D,
    Concatenate, Layer, BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from Bio import SeqIO

logger = logging.getLogger(__name__)

# ── Dictionaries ──────────────────────────────────────────────────────────────
# Mapping of the 20 standard amino acids to integer indices
AA_DICT = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

# Net charge contribution of each ionisable amino acid at physiological pH
CHARGE_DICT = {'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}

# Monoisotopic molecular mass (Da) for each amino acid residue
MASS_DICT = {'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
             'G': 75.07, 'H': 155.15, 'I': 131.17, 'K': 146.19, 'L': 131.17,
             'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
             'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}

# Kyte-Doolittle hydrophobicity scale for each amino acid
HYDRO_DICT = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
              'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
              'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
              'W': -0.9, 'Y': -1.3}


# ── Physicochemical Properties ────────────────────────────────────────────────

def compute_pI(sequence: str) -> float:
    """
    Estimates the isoelectric point (pI) of a peptide using a bisection method.
    Runs up to 60 iterations to find the pH at which the net charge is ~0.
    Accounts for N-terminus, C-terminus, and ionisable side chains (K, R, H, D, E).
    """
    pKa = {'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.3,
           'Nterm': 8.0, 'Cterm': 3.1}
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        # Contributions from terminal groups
        charge = (1 / (1 + 10 ** (mid - pKa['Nterm'])) -
                  1 / (1 + 10 ** (pKa['Cterm'] - mid)))
        # Contributions from ionisable side chains
        for aa in sequence:
            if aa in ('K', 'R', 'H'):
                charge += 1 / (1 + 10 ** (mid - pKa[aa]))
            elif aa in ('D', 'E'):
                charge -= 1 / (1 + 10 ** (pKa[aa] - mid))
        if abs(charge) < 1e-5:   # convergence criterion reached
            return mid
        if charge > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def raw_properties(sequence: str) -> np.ndarray:
    """
    Computes four physicochemical descriptors for a peptide sequence:
        [net charge, pI, molecular mass (Da), cumulative hydrophobicity].
    Returns a float32 array of shape (4,).
    Unknown residues receive neutral/average default values.
    """
    charge = sum(CHARGE_DICT.get(aa, 0) for aa in sequence)
    mass   = sum(MASS_DICT.get(aa, 110.0) for aa in sequence)   # 110 Da fallback
    hydro  = sum(HYDRO_DICT.get(aa, 0.0) for aa in sequence)
    pI     = compute_pI(sequence)
    return np.array([charge, pI, mass, hydro], dtype=np.float32)


# ── Normalizer ────────────────────────────────────────────────────────────────

class Normalizer:
    """
    Min-max scaler fitted on the training property matrix.
    Constant features are handled by artificially widening the range by ±1
    to avoid division by zero.
    """

    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, props):
        """Compute and store per-feature min and max from the training data."""
        self.mins = props.min(axis=0)
        self.maxs = props.max(axis=0)
        # Prevent division by zero for constant features
        eq = self.maxs == self.mins
        self.maxs[eq] += 1
        self.mins[eq] -= 1

    def transform(self, props):
        """Apply min-max normalisation using the stored statistics."""
        return ((props - self.mins) / (self.maxs - self.mins)).astype(np.float32)

    def fit_transform(self, props):
        """Fit on the data and then normalise it in one step."""
        self.fit(props)
        return self.transform(props)


# ── Custom Keras Layers ───────────────────────────────────────────────────────

class SamplingLayer(Layer):
    """
    Reparameterisation trick layer for the VAE latent space.
    Samples z = z_mean + eps * exp(0.5 * z_log_var),  eps ~ N(0, I).
    Allows gradients to flow through the stochastic sampling operation
    during back-propagation.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))   # standard normal noise
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAELossLayer(Layer):
    """
    Injects the VAE objective (ELBO) into the Keras computation graph.
    Compatible with Keras 3 via add_loss() and per-metric trackers.

    Total loss = reconstruction_loss + kl_weight * kl_loss
    where:
        reconstruction_loss = mean over batch of sum-of-squared-errors per position
        kl_loss             = -0.5 * mean( 1 + log_var - mean^2 - exp(log_var) )
    """

    def __init__(self, kl_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight   # beta coefficient scaling the KL term

        # Running-mean trackers exposed to the Keras logging infrastructure
        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker    = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x_true, x_recon, z_mean, z_log_var = inputs

        # Pixel-wise squared error summed across positions and AAs, averaged over batch
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x_true - x_recon), axis=[1, 2])
        )

        # Closed-form KL divergence from N(z_mean, exp(z_log_var)) to N(0, I)
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=-1
            )
        )

        total_loss = recon_loss + self.kl_weight * kl_loss
        self.add_loss(total_loss)   # register combined loss with the Keras optimizer

        # Update running averages for progress-bar display
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)

        return x_recon

    @property
    def metrics(self):
        """Expose reconstruction and KL sub-losses to Keras callbacks."""
        return [self.recon_tracker, self.kl_tracker]


# ── CVAE Model ────────────────────────────────────────────────────────────────

class CVAEModel:
    """
    Conditional Variational Autoencoder (CVAE) for antimicrobial peptide (AMP) design.

    Architecture overview:
        Encoder : Conv1D stack → Flatten → Concatenate(condition) → Dense
                  → z_mean, z_log_var → SamplingLayer → z
        Decoder : Concatenate(z, condition) → Dense → Reshape → Conv1D (softmax)
        Full VAE: trained end-to-end; loss injected via VAELossLayer.

    Args:
        large_fasta : Path to the large FASTA database used for pre-training.
        max_length  : Maximum peptide length; shorter sequences are zero-padded.
        latent_dim  : Dimensionality of the latent space z.
        kl_weight   : Beta coefficient balancing KL divergence vs reconstruction.
    """

    def __init__(self, large_fasta, max_length=50,
                 latent_dim=128, kl_weight=0.5):
        self.large_fasta = large_fasta
        self.max_length  = max_length
        self.latent_dim  = latent_dim
        self.kl_weight   = kl_weight
        self.aa_dict     = AA_DICT
        self.num_aa      = len(AA_DICT)
        self.normalizer  = Normalizer()

        # Model sub-components — populated by build_model()
        self.model   = None
        self.encoder = None
        self.decoder = None

    def _one_hot(self, seq):
        """
        Encode a peptide sequence as a (max_length, num_aa) one-hot matrix.
        Positions beyond the sequence length remain zero (padding).
        """
        enc = np.zeros((self.max_length, self.num_aa), dtype=np.float32)
        for i, aa in enumerate(seq[:self.max_length]):
            if aa in self.aa_dict:
                enc[i, self.aa_dict[aa]] = 1.0
        return enc

    def _clean(self, seq):
        """Strip non-standard residues and convert to uppercase."""
        return ''.join(a for a in seq.upper() if a in self.aa_dict)

    def load_data(self):
        """
        Parse the large FASTA file, encode sequences and normalise conditions.

        Returns:
            X     : np.ndarray (N, max_length, num_aa) — one-hot encoded sequences.
            props : np.ndarray (N, 4)                  — normalised property vectors.
        """
        records = list(SeqIO.parse(self.large_fasta, 'fasta'))
        seqs    = [self._clean(str(r.seq)) for r in records]
        seqs    = [s for s in seqs if len(s) >= 4]  # discard very short peptides

        if len(seqs) == 0:
            raise ValueError("No valid sequences found in the FASTA file.")

        logger.info(f"{len(seqs)} valid sequences loaded from '{self.large_fasta}'.")

        X     = np.array([self._one_hot(s) for s in seqs])
        props = np.array([raw_properties(s) for s in seqs])
        props = self.normalizer.fit_transform(props)  # fit scaler on training set

        return X, props

    def build_model(self):
        """
        Build the encoder, decoder and full CVAE graph; compile with Adam.
        The VAELossLayer registers the ELBO loss — no explicit loss= argument needed.
        """
        cond_dim = 4  # condition vector: [charge, pI, mass, hydrophobicity]

        # ── Encoder ──────────────────────────────────────────────────────────
        inp_seq  = Input(shape=(self.max_length, self.num_aa))  # one-hot sequence
        inp_cond = Input(shape=(cond_dim,))                     # physicochemical condition

        # Hierarchical convolutional feature extraction
        x = Conv1D(64, 3, activation='relu', padding='same')(inp_seq)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Flatten()(x)

        # Merge sequence features with the condition vector
        x = Concatenate()([x, inp_cond])
        x = Dense(256, activation='relu')(x)

        # Latent distribution parameters
        z_mean    = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z         = SamplingLayer()([z_mean, z_log_var])  # reparameterised sample

        self.encoder = Model([inp_seq, inp_cond],
                             [z_mean, z_log_var, z])

        # ── Decoder ──────────────────────────────────────────────────────────
        inp_z      = Input(shape=(self.latent_dim,))  # latent vector
        inp_cond_d = Input(shape=(cond_dim,))         # condition vector (decoder side)

        h = Concatenate()([inp_z, inp_cond_d])
        h = Dense(256, activation='relu')(h)
        h = Dense(self.max_length * 64, activation='relu')(h)
        h = Reshape((self.max_length, 64))(h)
        # Softmax over the amino-acid vocabulary at each sequence position
        h = Conv1D(self.num_aa, 3, activation='softmax', padding='same')(h)

        self.decoder = Model([inp_z, inp_cond_d], h)

        # ── Full VAE (encoder → decoder → loss) ──────────────────────────────
        z_mean, z_log_var, z_s = self.encoder([inp_seq, inp_cond])
        x_recon = self.decoder([z_s, inp_cond])

        # Wrap the reconstruction with the ELBO loss layer
        x_recon = VAELossLayer(self.kl_weight)(
            [inp_seq, x_recon, z_mean, z_log_var]
        )

        self.model = Model([inp_seq, inp_cond], x_recon)
        self.model.compile(optimizer=Adam(1e-3))  # loss handled by add_loss()

        logger.info("CVAE model built successfully.")

    def train(self, X, props, epochs=100, batch_size=32):
        """
        Train the CVAE end-to-end.

        Callbacks used:
            EarlyStopping     — halts training if val_loss does not improve for 10 epochs.
            ReduceLROnPlateau — reduces learning rate by 10x after 5 stagnant epochs.
            CSVLogger         — saves per-epoch metrics to 'training_log.csv'.
        """
        if self.model is None:
            self.build_model()

        self.model.fit(
            [X, props], X,          # reconstruction target is the input sequence
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,   # hold out 10% of data for validation
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5),
                CSVLogger("training_log.csv")
            ],
            verbose=1
        )

    def generate(self, condition, latent_sample=None, temperature=1.0):
        """
        Decode a latent vector + condition into a per-position AA distribution.

        Args:
            condition     : Normalised condition vector, shape (4,).
            latent_sample : Optional fixed z vector, shape (1, latent_dim).
                            If None, drawn from N(0, I).
            temperature   : Sampling temperature.
                            < 1 → sharper / more deterministic output.
                            > 1 → softer / more diverse output.

        Returns:
            np.ndarray of shape (max_length, num_aa) — per-position AA probabilities.
        """
        if self.decoder is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if latent_sample is None:
            # Sample from the standard normal prior p(z)
            latent_sample = np.random.normal(
                size=(1, self.latent_dim)).astype(np.float32)

        cond = condition.reshape(1, -1).astype(np.float32)

        probs = self.decoder.predict([latent_sample, cond], verbose=0)[0]

        # Apply temperature rescaling when requested
        if temperature != 1.0:
            logits  = np.log(np.clip(probs, 1e-8, 1.0)) / temperature
            logits -= logits.max(axis=-1, keepdims=True)   # numerical stability
            probs   = np.exp(logits)
            probs  /= probs.sum(axis=-1, keepdims=True)

        return probs.astype(np.float32)

    def normalize_condition(self, props_raw):
        """
        Normalise a raw property vector using the scaler fitted in load_data().

        Args:
            props_raw : array of shape (4,) — [charge, pI, mass, hydrophobicity].

        Returns:
            Normalised float32 array of shape (4,).
        """
        return self.normalizer.transform(props_raw.reshape(1, -1))[0]
