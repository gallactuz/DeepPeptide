
"""
lstm_model.py — LSTM autoregressivo condicional para refinamento de AMPs.

Condicionamento v3:
  - [charge, pI, mass, hydrophobicity, rank]
  - rank = posição normalizada no arquivo (0.0 = menos potente, 1.0 = mais potente)
  - Funciona para qualquer banco ordenado: MIC crescente, IC50, atividade, etc.
  - Na geração: target_potency=1.0 direciona para sequências do topo do arquivo.
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
    LSTM autoregressivo condicional.

    O banco pequeno é assumido estar ORDENADO do menos potente ao mais potente
    (ex.: MIC decrescente → potência crescente). A posição normalizada no arquivo
    [0.0, 1.0] é usada como 5ª dimensão de condição (rank de potência).

    Na geração, target_potency=1.0 gera sequências próximas ao topo da lista.
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
        self._sequences   = None
        self.n_sequences  = 0         # exposto para ponderação externa

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _clean(self, seq: str) -> str:
        return ''.join(aa for aa in seq.upper() if aa in self.aa_dict)

    def _one_hot(self, seq: str) -> np.ndarray:
        enc = np.zeros((self.max_length, self.num_aa), dtype=np.float32)
        for i, aa in enumerate(seq[:self.max_length]):
            if aa in self.aa_dict:
                enc[i, self.aa_dict[aa]] = 1.0
        return enc

    # ── Dados ─────────────────────────────────────────────────────────────────

    def load_data(self):
        """
        Carrega sequências e prepara pares autoregressivos (X, y, cond).

        cond = [charge_norm, pI_norm, mass_norm, hydro_norm, rank_norm]
        rank_norm = índice / (N-1)  →  0.0 (início do arquivo) … 1.0 (fim)
        """
        records = list(SeqIO.parse(self.small_fasta, 'fasta'))
        seqs    = [self._clean(str(r.seq)) for r in records]
        seqs    = [s for s in seqs if len(s) >= 4]

        if len(seqs) < 2:
            raise ValueError(
                f"'{self.small_fasta}' precisa de ao menos 2 sequências válidas.")

        self._sequences  = seqs
        self.n_sequences = len(seqs)
        n = len(seqs)

        # rank ordinal normalizado: 0.0 = menos potente, 1.0 = mais potente
        ranks = np.linspace(0.0, 1.0, n, dtype=np.float32)  # shape (N,)

        logger.info(
            f"LSTM ({self.small_fasta}): {n} seqs | "
            f"rank 0.0 → {seqs[0][:8]}... (menos potente) | "
            f"rank 1.0 → {seqs[-1][:8]}... (mais potente)")

        # propriedades físico-químicas
        props_raw = np.array([raw_properties(s) for s in seqs], dtype=np.float32)
        props_norm = self._local_normalize(props_raw)  # (N, 4)

        # concatena rank como 5ª dimensão
        cond_full = np.column_stack([props_norm, ranks])  # (N, 5)

        # pares autoregressivos
        X_list, y_list, cond_list = [], [], []
        for seq, cond in zip(seqs, cond_full):
            oh = self._one_hot(seq)                           # (max_length, 20)
            X_list.append(oh[:-1])                           # tokens 0…L-2
            y_list.append(oh[1:])                            # tokens 1…L-1
            cond_list.append(
                np.tile(cond, (self.max_length - 1, 1)))     # (T-1, 5)

        X    = np.array(X_list,    dtype=np.float32)   # (N, T-1, 20)
        y    = np.array(y_list,    dtype=np.float32)   # (N, T-1, 20)
        cond = np.array(cond_list, dtype=np.float32)   # (N, T-1, 5)
        return X, y, cond

    # ── Normalização local ────────────────────────────────────────────────────

    def _local_normalize(self, props: np.ndarray) -> np.ndarray:
        """Min-max local para as 4 propriedades físico-químicas (SEM o rank)."""
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
        Normaliza propriedades + injeta rank de potência alvo.

        Args:
            props_raw:       array (4,) — [charge, pI, mass, hydrophobicity]
            target_potency:  float [0.0, 1.0]
                             0.0 = geração próxima ao início do arquivo (menos potente)
                             1.0 = geração próxima ao fim do arquivo (mais potente)

        Returns:
            array (5,) normalizado.
        """
        props_4n = ((props_raw - self._local_mins) /
                    (self._local_maxs - self._local_mins)).astype(np.float32)
        rank     = np.clip(target_potency, 0.0, 1.0)
        return np.append(props_4n, rank).astype(np.float32)

    # ── Seed por faixa de potência ────────────────────────────────────────────

    def seed_by_potency(self, target_potency: float = 1.0,
                        window: float = 0.2) -> str:
        """
        Retorna uma sequência aleatória da faixa de potência desejada.

        Ex.: target_potency=1.0, window=0.2 → sorteia entre os 20% finais do arquivo.
        Ex.: target_potency=0.5, window=0.2 → sorteia entre 40%–60% do arquivo.
        """
        if self._sequences is None:
            raise RuntimeError("Chame load_data() antes.")
        n     = len(self._sequences)
        lo    = max(0, int((target_potency - window / 2) * n))
        hi    = min(n, int((target_potency + window / 2) * n) + 1)
        if lo >= hi:
            lo, hi = max(0, hi - 1), hi
        idx = np.random.randint(lo, hi)
        return self._sequences[idx]

    def random_seed(self) -> str:
        if self._sequences is None:
            raise RuntimeError("Chame load_data() antes.")
        return self._sequences[np.random.randint(len(self._sequences))]

    def last_sequence(self) -> str:
        if self._sequences is None:
            raise RuntimeError("Chame load_data() antes.")
        return self._sequences[-1]

    # ── Arquitetura ───────────────────────────────────────────────────────────

    def build_model(self):
        inp_seq  = Input(shape=(None, self.num_aa),   name='lstm_seq')
        inp_cond = Input(shape=(None, self.cond_dim), name='lstm_cond')  # (T, 5)

        x   = Concatenate(axis=-1)([inp_seq, inp_cond])
        x   = Masking(mask_value=0.0)(x)
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
        logger.info(f"LSTM construído (cond_dim={self.cond_dim} = "
                    f"charge + pI + mass + hydro + rank).")
        self.model.summary(print_fn=logger.info)

    # ── Treino ────────────────────────────────────────────────────────────────

    def train(self, X, y, cond, epochs=200, batch_size=16):
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

    # ── Geração autoregressiva ────────────────────────────────────────────────

    def generate(self, target_length: int,
                 condition: np.ndarray,
                 temperature: float = 1.0,
                 target_potency: float = 1.0) -> np.ndarray:
        """
        Gera distribuição de probabilidades por posição (autoregressivo).

        A seed é escolhida da faixa de potência correspondente a target_potency,
        garantindo que o contexto inicial já seja coerente com a potência desejada.

        Args:
            target_length:   número de aminoácidos a gerar.
            condition:       array (5,) normalizado (inclui rank).
            temperature:     diversidade de amostragem.
            target_potency:  [0.0, 1.0] — de onde sortear a seed no arquivo.

        Returns:
            np.ndarray (target_length, num_aa).
        """
        if self.model is None:
            raise RuntimeError("Execute train() antes de generate().")

        aa_list     = list(self.aa_dict.keys())
        seed        = self.seed_by_potency(target_potency)   # seed coerente
        current_seq = list(seed[:self.max_length])
        all_probs   = []

        for _ in range(target_length):
            inp_oh   = self._one_hot(''.join(current_seq))[np.newaxis]       # (1,T,20)
            inp_cond = np.tile(condition, (1, inp_oh.shape[1], 1)).astype(np.float32)

            probs = self.model.predict([inp_oh, inp_cond], verbose=0)[0][-1]  # (20,)

            if temperature != 1.0:
                logits  = np.log(np.clip(probs, 1e-8, 1.0)) / temperature
                logits -= logits.max()
                probs   = np.exp(logits)
                probs  /= probs.sum()

            all_probs.append(probs.copy())
            next_idx = np.random.choice(self.num_aa, p=probs)
            current_seq.append(aa_list[next_idx])
            if len(current_seq) > self.max_length:
                current_seq = current_seq[-self.max_length:]

        return np.array(all_probs, dtype=np.float32)   # (target_length, 20)

