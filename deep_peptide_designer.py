
"""
deep_peptide_designer.py — Orquestrador CVAE + LSTMs para design de AMPs.

Uso básico:
    python deep_peptide_designer.py \
        -vae bigdatabase.fasta \
        -lstm mic_database.fasta \
        --length 15 --num 10 \
        --target_potency 1.0      # ← gera próximo dos mais potentes do banco

Múltiplos bancos pequenos:
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
    Combina CVAE (banco grande) + N LSTMs ordenados por potência (bancos pequenos).

    Premissa dos bancos pequenos:
        As sequências estão ORDENADAS do menos potente ao mais potente
        (ex.: MIC decrescente → potência crescente).
        A posição normalizada [0, 1] é usada como condição de potência (rank).

    Fluxo de geração:
        1. CVAE: z ~ N(0,I) → P_vae (distribuição base, max_length × 20).
        2. Cada LSTM: seed da faixa de potência alvo → P_lstm_i (target_length × 20).
        3. _combine(): Product of Experts ponderado por tamanho de banco.
        4. Filtros de propriedades físico-químicas.
    """

    def __init__(self, vae_fasta: str, lstm_fastas: list,
                 alpha: float = 0.5, max_length: int = 50,
                 latent_dim: int = 128, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Seed de reprodutibilidade: {seed}")

        self.alpha      = alpha
        self.max_length = max_length
        self.aa_dict    = AA_DICT
        self.aa_list    = list(AA_DICT.keys())
        self.num_aa     = len(AA_DICT)

        self.cvae  = CVAEModel(vae_fasta, max_length=max_length,
                               latent_dim=latent_dim)
        self.lstms = [LSTMModel(f, max_length=max_length) for f in lstm_fastas]
        self._lstm_weights = None

    # ── Inicialização ─────────────────────────────────────────────────────────

    def initialize(self, vae_epochs=100, lstm_epochs=200, batch_size=32):
        logger.info("── Carregando e treinando CVAE ──")
        X_vae, props_vae = self.cvae.load_data()
        self.cvae.train(X_vae, props_vae, epochs=vae_epochs, batch_size=batch_size)

        logger.info("── Carregando e treinando LSTMs ──")
        sizes = []
        for i, lstm in enumerate(self.lstms):
            logger.info(f"LSTM {i+1}/{len(self.lstms)}: {lstm.small_fasta}")
            X, y, cond = lstm.load_data()
            lstm.train(X, y, cond, epochs=lstm_epochs, batch_size=batch_size)
            sizes.append(lstm.n_sequences)

        sizes = np.array(sizes, dtype=np.float32)
        self._lstm_weights = sizes / sizes.sum() if sizes.sum() > 0 else \
                             np.ones(len(sizes)) / len(sizes)

        for i, (lstm, w) in enumerate(zip(self.lstms, self._lstm_weights)):
            logger.info(f"  LSTM {i+1} ({Path(lstm.small_fasta).name}): "
                        f"{lstm.n_sequences} seqs → peso {w:.3f}")
        logger.info("Todos os modelos treinados.")

    # ── Propriedades ──────────────────────────────────────────────────────────

    @staticmethod
    def calculate_properties(sequence: str) -> dict:
        props = raw_properties(sequence)
        return {
            'charge':         float(props[0]),
            'pI':             float(props[1]),
            'mass':           float(props[2]),
            'hydrophobicity': float(props[3]),
            'length':         len(sequence),
        }

    # ── Combinação ────────────────────────────────────────────────────────────

    def _combine(self, p_vae: np.ndarray,
                 p_lstms: list,
                 target_length: int) -> np.ndarray:
        """
        Product of Experts ponderado:
            log P = α·log P_vae + (1−α)·Σ w_i·log P_lstm_i
        """
        p_vae_crop = p_vae[:target_length]

        if p_lstms and self._lstm_weights is not None:
            weights     = self._lstm_weights[:len(p_lstms)]
            weights     = weights / weights.sum()
            log_p_lstm  = np.zeros_like(p_lstms[0])
            for w, p_l in zip(weights, p_lstms):
                ml = min(p_vae_crop.shape[0], p_l.shape[0])
                log_p_lstm[:ml] += w * np.log(np.clip(p_l[:ml], 1e-8, 1.0))

            ml           = min(p_vae_crop.shape[0], log_p_lstm.shape[0])
            log_p_vae    = np.log(np.clip(p_vae_crop[:ml], 1e-8, 1.0))
            combined_log = self.alpha * log_p_vae + (1 - self.alpha) * log_p_lstm[:ml]
            combined     = np.exp(combined_log)
        else:
            combined = p_vae_crop

        sums     = combined.sum(axis=-1, keepdims=True)
        combined = combined / np.where(sums == 0, 1, sums)
        return combined

    # ── Amostragem ────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_sequence(probs: np.ndarray, aa_list: list,
                         temperature: float = 1.0) -> str:
        sequence = []
        for dist in probs:
            if temperature != 1.0:
                logits  = np.log(np.clip(dist, 1e-8, 1.0)) / temperature
                logits -= logits.max()
                dist    = np.exp(logits)
                dist   /= dist.sum()
            idx = np.random.choice(len(aa_list), p=dist)
            sequence.append(aa_list[idx])
        return ''.join(sequence)

    # ── Design de peptídeos ───────────────────────────────────────────────────

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
        Gera AMPs candidatos.

        Args:
            target_potency: [0.0, 1.0] — posição alvo no ranking dos bancos pequenos.
                            1.0 = gerar próximo das sequências mais potentes (topo do arquivo).
                            0.5 = região mediana.
                            0.0 = menos potentes (raramente útil).
        """
        # ── condição CVAE ─────────────────────────────────────────────────
        raw_cond = np.array([
            (target_properties or {}).get('charge',         2.0),
            (target_properties or {}).get('pI',             8.0),
            (target_properties or {}).get('mass',        1500.0),
            (target_properties or {}).get('hydrophobicity', 5.0),
        ], dtype=np.float32)
        norm_cond_cvae = self.cvae.normalize_condition(raw_cond)

        logger.info(f"Potência alvo nos LSTMs: {target_potency:.2f} "
                    f"({'mais potente' if target_potency >= 0.8 else 'moderado' if target_potency >= 0.4 else 'menos potente'})")

        # ── condição de cada LSTM (rank = target_potency) ─────────────────
        lstm_conds = []
        for lstm in self.lstms:
            seed_props = raw_properties(lstm.seed_by_potency(target_potency))
            lstm_conds.append(
                lstm.normalize_condition(seed_props, target_potency=target_potency)
            )

        peptides = []
        for idx in range(num_peptides):
            best = None
            best_props = None

            for _ in range(max_attempts):
                z = np.random.normal(
                    size=(1, self.cvae.latent_dim)).astype(np.float32)
                p_vae = self.cvae.generate(norm_cond_cvae,
                                           latent_sample=z,
                                           temperature=temperature)

                p_lstms = []
                for lstm, cond in zip(self.lstms, lstm_conds):
                    p_l = lstm.generate(
                        target_length  = target_length,
                        condition      = cond,
                        temperature    = temperature,
                        target_potency = target_potency   # seed da faixa certa
                    )
                    p_lstms.append(p_l)

                combined   = self._combine(p_vae, p_lstms, target_length)
                seq        = self._sample_sequence(combined, self.aa_list, temperature)
                props      = self.calculate_properties(seq)
                best       = seq
                best_props = props

                if (props['charge'] >= min_charge and
                        props['hydrophobicity'] >= min_hydrophobicity):
                    break
            else:
                logger.warning(
                    f"Peptídeo #{idx+1}: critérios não atendidos após "
                    f"{max_attempts} tentativas — melhor candidato aceito.")

            entry = {'sequence': best, **best_props}
            peptides.append(entry)
            logger.info(
                f"AMP #{idx+1}: {best} | "
                f"charge={best_props['charge']:.2f} "
                f"pI={best_props['pI']:.2f} "
                f"mass={best_props['mass']:.1f} "
                f"hydro={best_props['hydrophobicity']:.2f}")

        return peptides


# ── Saída ──────────────────────────────────────────────────────────────────────

def save_results(peptides: list, output_path: str):
    p = Path(output_path)
    with open(p, 'w', newline='') as f:
        fields = ['sequence', 'charge', 'pI', 'mass', 'hydrophobicity', 'length']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(peptides)
    logger.info(f"CSV salvo: {p}")

    fasta_path = p.with_suffix('.fasta')
    with open(fasta_path, 'w') as f:
        for i, pep in enumerate(peptides, 1):
            f.write(f">AMP_{i} charge={pep['charge']:.2f} pI={pep['pI']:.2f} "
                    f"mass={pep['mass']:.1f} hydro={pep['hydrophobicity']:.2f}\n")
            f.write(pep['sequence'] + '\n')
    logger.info(f"FASTA salvo: {fasta_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Design de AMPs com CVAE + LSTM condicional por potência.')

    parser.add_argument('-vae',  required=True,
                        help='FASTA grande para o CVAE.')
    parser.add_argument('-lstm', action='append', required=True,
                        help='FASTA(s) pequeno(s) ORDENADO(s) por potência crescente.')

    parser.add_argument('--alpha',           type=float, default=0.5,
                        help='Peso CVAE vs LSTM (0=só LSTM, 1=só CVAE).')
    parser.add_argument('--length',          type=int,   default=15)
    parser.add_argument('--num',             type=int,   default=5)
    parser.add_argument('--temperature',     type=float, default=1.0)
    parser.add_argument('--max_length',      type=int,   default=50)

    parser.add_argument('--min_charge',         type=float, default=1.0)
    parser.add_argument('--min_hydrophobicity', type=float, default=0.0)
    parser.add_argument('--max_attempts',       type=int,   default=50)

    # ── Potência alvo ─────────────────────────────────────────────────────
    parser.add_argument('--target_potency',  type=float, default=1.0,
                        help='[0.0–1.0] Região do ranking de potência a imitar. '
                             '1.0 = mais potente (padrão). '
                             'Reflete a posição no arquivo FASTA ordenado.')

    # ── Propriedades físico-químicas alvo (CVAE) ───────────────────────────
    parser.add_argument('--target_charge',         type=float, default=None)
    parser.add_argument('--target_pI',             type=float, default=None)
    parser.add_argument('--target_mass',           type=float, default=None)
    parser.add_argument('--target_hydrophobicity', type=float, default=None)

    # ── Treino ────────────────────────────────────────────────────────────
    parser.add_argument('--vae_epochs',  type=int, default=100)
    parser.add_argument('--lstm_epochs', type=int, default=200)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--latent_dim',  type=int, default=128)

    parser.add_argument('--seed',   type=int, default=None)
    parser.add_argument('--output', type=str, default='designed_amps.csv')

    args = parser.parse_args()

    target_props = None
    if any(v is not None for v in [args.target_charge, args.target_pI,
                                    args.target_mass, args.target_hydrophobicity]):
        target_props = {
            'charge':         args.target_charge         or 2.0,
            'pI':             args.target_pI             or 8.0,
            'mass':           args.target_mass           or 1500.0,
            'hydrophobicity': args.target_hydrophobicity or 5.0,
        }
        logger.info(f"Propriedades-alvo (CVAE): {target_props}")

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

    print("\n" + "═" * 65)
    print(f"{'#':>3}  {'Sequência':<25} {'Charge':>7} {'pI':>6} "
          f"{'Mass':>8} {'Hydro':>7}")
    print("─" * 65)
    for i, p in enumerate(peptides, 1):
        print(f"{i:>3}  {p['sequence']:<25} {p['charge']:>7.2f} "
              f"{p['pI']:>6.2f} {p['mass']:>8.1f} {p['hydrophobicity']:>7.2f}")
    print("═" * 65)

    save_results(peptides, args.output)


if __name__ == '__main__':
    main()

