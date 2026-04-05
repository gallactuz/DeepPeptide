"""
cvae_model.py — Conditional Variational Autoencoder para design de AMPs.
Compatível com Keras 3.
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

# ── Dicionários ───────────────────────────────────────────────────────────────
AA_DICT = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
CHARGE_DICT = {'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}
MASS_DICT = {'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
             'G': 75.07, 'H': 155.15, 'I': 131.17, 'K': 146.19, 'L': 131.17,
             'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
             'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}
HYDRO_DICT = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
              'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
              'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
              'W': -0.9, 'Y': -1.3}


# ── Propriedades ──────────────────────────────────────────────────────────────
def compute_pI(sequence: str) -> float:
    pKa = {'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.3,
           'Nterm': 8.0, 'Cterm': 3.1}
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        charge = (1 / (1 + 10 ** (mid - pKa['Nterm'])) -
                  1 / (1 + 10 ** (pKa['Cterm'] - mid)))
        for aa in sequence:
            if aa in ('K', 'R', 'H'):
                charge += 1 / (1 + 10 ** (mid - pKa[aa]))
            elif aa in ('D', 'E'):
                charge -= 1 / (1 + 10 ** (pKa[aa] - mid))
        if abs(charge) < 1e-5:
            return mid
        if charge > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def raw_properties(sequence: str) -> np.ndarray:
    charge = sum(CHARGE_DICT.get(aa, 0) for aa in sequence)
    mass = sum(MASS_DICT.get(aa, 110.0) for aa in sequence)
    hydro = sum(HYDRO_DICT.get(aa, 0.0) for aa in sequence)
    pI = compute_pI(sequence)
    return np.array([charge, pI, mass, hydro], dtype=np.float32)


# ── Normalizador ──────────────────────────────────────────────────────────────
class Normalizer:
    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, props):
        self.mins = props.min(axis=0)
        self.maxs = props.max(axis=0)
        eq = self.maxs == self.mins
        self.maxs[eq] += 1
        self.mins[eq] -= 1

    def transform(self, props):
        return ((props - self.mins) / (self.maxs - self.mins)).astype(np.float32)

    def fit_transform(self, props):
        self.fit(props)
        return self.transform(props)


# ── Camadas ───────────────────────────────────────────────────────────────────
class SamplingLayer(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAELossLayer(Layer):
    """Compatível com Keras 3"""

    def __init__(self, kl_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight

        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x_true, x_recon, z_mean, z_log_var = inputs

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x_true - x_recon), axis=[1, 2])
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=-1
            )
        )

        total_loss = recon_loss + self.kl_weight * kl_loss
        self.add_loss(total_loss)

        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)

        return x_recon

    @property
    def metrics(self):
        return [self.recon_tracker, self.kl_tracker]


# ── Modelo ────────────────────────────────────────────────────────────────────
class CVAEModel:

    def __init__(self, large_fasta, max_length=50,
                 latent_dim=128, kl_weight=0.5):
        self.large_fasta = large_fasta
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.aa_dict = AA_DICT
        self.num_aa = len(AA_DICT)
        self.normalizer = Normalizer()

        # 🔥 IMPORTANTE
        self.model = None
        self.encoder = None
        self.decoder = None

    def _one_hot(self, seq):
        enc = np.zeros((self.max_length, self.num_aa), dtype=np.float32)
        for i, aa in enumerate(seq[:self.max_length]):
            if aa in self.aa_dict:
                enc[i, self.aa_dict[aa]] = 1.0
        return enc

    def _clean(self, seq):
        return ''.join(a for a in seq.upper() if a in self.aa_dict)

    def load_data(self):
        records = list(SeqIO.parse(self.large_fasta, 'fasta'))
        seqs = [self._clean(str(r.seq)) for r in records]
        seqs = [s for s in seqs if len(s) >= 4]

        if len(seqs) == 0:
            raise ValueError("Nenhuma sequência válida.")

        logger.info(f"{len(seqs)} sequências")

        X = np.array([self._one_hot(s) for s in seqs])
        props = np.array([raw_properties(s) for s in seqs])
        props = self.normalizer.fit_transform(props)

        return X, props

    def build_model(self):
        cond_dim = 4

        # ── Encoder ──
        inp_seq = Input(shape=(self.max_length, self.num_aa))
        inp_cond = Input(shape=(cond_dim,))

        x = Conv1D(64, 3, activation='relu', padding='same')(inp_seq)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Flatten()(x)

        x = Concatenate()([x, inp_cond])
        x = Dense(256, activation='relu')(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = SamplingLayer()([z_mean, z_log_var])

        self.encoder = Model([inp_seq, inp_cond],
                             [z_mean, z_log_var, z])

        # ── Decoder ──
        inp_z = Input(shape=(self.latent_dim,))
        inp_cond_d = Input(shape=(cond_dim,))

        h = Concatenate()([inp_z, inp_cond_d])
        h = Dense(256, activation='relu')(h)
        h = Dense(self.max_length * 64, activation='relu')(h)
        h = Reshape((self.max_length, 64))(h)
        h = Conv1D(self.num_aa, 3, activation='softmax', padding='same')(h)

        self.decoder = Model([inp_z, inp_cond_d], h)

        # ── VAE completo ──
        z_mean, z_log_var, z_s = self.encoder([inp_seq, inp_cond])
        x_recon = self.decoder([z_s, inp_cond])

        x_recon = VAELossLayer(self.kl_weight)(
            [inp_seq, x_recon, z_mean, z_log_var]
        )

        self.model = Model([inp_seq, inp_cond], x_recon)
        self.model.compile(optimizer=Adam(1e-3))

        logger.info("CVAE construído.")

    def train(self, X, props, epochs=100, batch_size=32):
        if self.model is None:
            self.build_model()

        self.model.fit(
            [X, props], X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5), 
                CSVLogger("training_log.csv")
            ],
            verbose=1
        )

    def generate(self, condition, latent_sample=None, temperature=1.0):
        if self.decoder is None:
            raise RuntimeError("Modelo não treinado.")

        if latent_sample is None:
            latent_sample = np.random.normal(size=(1, self.latent_dim)).astype(np.float32)

        cond = condition.reshape(1, -1).astype(np.float32)

        probs = self.decoder.predict([latent_sample, cond], verbose=0)[0]

        if temperature != 1.0:
            logits = np.log(np.clip(probs, 1e-8, 1.0)) / temperature
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)

        return probs.astype(np.float32)

    def normalize_condition(self, props_raw):
        return self.normalizer.transform(props_raw.reshape(1, -1))[0]
