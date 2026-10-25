#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo - Distinguisher neuronal para o SHA-256 com rondas reduzidas (CUDA)
======================================================================
"""

import argparse
import struct
import sys
import numpy as np

# Substituição do scikit-learn pelo PyTorch para aceleração via GPU
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# SHA-256 com numero de rondas configuravel (Constantes permanecem na CPU)
# ---------------------------------------------------------------------------
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

MASK = 0xffffffff


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def sha256_block_rounds(block64, rounds):
    w = list(struct.unpack('>16I', block64))
    for i in range(16, 64):
        s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & MASK)

    a, b, c, d, e, f, g, h = H0
    for i in range(rounds):
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = (h + S1 + ch + K[i] + w[i]) & MASK
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & MASK
        h, g, f = g, f, e
        e = (d + t1) & MASK
        d, c, b = c, b, a
        a = (t1 + t2) & MASK

    out = [(H0[i] + v) & MASK for i, v in enumerate([a, b, c, d, e, f, g, h])]
    return struct.pack('>8I', *out)


def make_delta(nbytes=64, bit=0):
    d = bytearray(nbytes)
    d[bit // 8] = 1 << (7 - (bit % 8))
    return bytes(d)


# ---------------------------------------------------------------------------
# Geração Otimizada de Dataset com Destino à GPU
# ---------------------------------------------------------------------------
def make_dataset_cuda(n, rounds, delta, rng, device):
    """Gera n amostras e transfere os vetores de bits binários para o hardware CUDA."""
    X_cpu = np.zeros((n, 64), dtype=np.uint8) # 64 bytes = 512 bits combinados (C0 + C1)
    y_cpu = np.zeros(n, dtype=np.float32)     # Alvos em float32 para as funções de perda do PyTorch
    
    for i in range(n):
        m0 = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
        c0 = sha256_block_rounds(m0, rounds)
        label = i & 1
        if label == 1:
            m1 = bytes(x ^ z for x, z in zip(m0, delta))
            c1 = sha256_block_rounds(m1, rounds)
        else:
            mr = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
            c1 = sha256_block_rounds(mr, rounds)
            
        X_cpu[i] = np.frombuffer(c0 + c1, dtype=np.uint8)
        y_cpu[i] = float(label)
        
    # Descompactação massiva de bits (unpackbits) antes da transferência para CUDA
    X_bits = np.unpackbits(X_cpu, axis=1)
    
    # Criação dos Tensores nativos e envio para o Dispositivo Ativo
    X_tensor = torch.from_numpy(X_bits).float().to(device)
    y_tensor = torch.from_numpy(y_cpu).float().unsqueeze(1).to(device) # Shape (n, 1)
    
    return X_tensor, y_tensor


# ---------------------------------------------------------------------------
# Arquitetura da Rede Neuronal em PyTorch (Substitui o MLPClassifier)
# ---------------------------------------------------------------------------
class DistinguisherMLP(nn.Module):
    def __init__(self):
        super(DistinguisherMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Saída binária: probabilidade de ser Par Real (Classe 1)
        )
        
    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Pipeline de Execução da Curva com CUDA
# ---------------------------------------------------------------------------
def run_curve_cuda(rounds_list, n_train, n_test, delta_bit, seed):
    if not HAVE_TORCH:
        sys.exit("Instale o PyTorch para execução: pip install torch")

    # Deteta e ativa a GPU via CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print(f"Estudo 4 - Distinguisher neuronal SHA-256 (Modo: {device.type.upper()})")
    print("=" * 64)
    print(f"Treino = {n_train} amostras | Teste = {n_test} amostras (SEPARADAS)")
    print(f"DELTA = 1 bit na posicao {delta_bit} | semente = {seed}")
    print("-" * 64)
    print(f"{'Rondas':>7} | {'Exatidao teste':>15} | Leitura")
    print("-" * 64)

    delta = make_delta(bit=delta_bit)
    results = []

    for r in rounds_list:
        torch.manual_seed(seed)
        rng_tr = np.random.default_rng(seed + r)
        rng_te = np.random.default_rng(seed + 100000 + r)
        
        # Geração e envio imediato para CUDA
        Xtr, ytr = make_dataset_cuda(n_train, r, delta, rng_tr, device)
        Xte, yte = make_dataset_cuda(n_test, r, delta, rng_te, device)

        # Inicialização do Modelo e Otimizadores em GPU
        model = DistinguisherMLP().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # weight_decay = alpha

        # Configuração de treino com mini-lotes (Replicando o batch_size original)
        batch_size = 256
        epochs = 60
        
        # Implementação manual e simplificada de Early Stopping para performance
        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 6
        
        model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(Xtr.size(0))
            epoch_loss = 0.0
            
            for i in range(0, Xtr.size(0), batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = Xtr[indices], ytr[indices]
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            # Validação do Early Stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                break

        # Inferência e avaliação final na GPU
        model.eval()
        with torch.no_grad():
            preds_te = model(Xte)
            preds_binary = (preds_te > 0.5).float()
            # Cálculo de exatidão em hardware CUDA (evita cast de tipos inteiros usando .float())
            acc = float((preds_binary == yte).float().mean().item())

        results.append((r, acc))

        if acc >= 0.55:
            leitura = "ESTRUTURA detetavel (vulneravel)"
        elif acc >= 0.52:
            leitura = "vantagem residual"
        else:
            leitura = "~acaso (robusto)"
        print(f"{r:>7} | {acc*100:>13.2f}% | {leitura}")

    print("-" * 64)
    print("50% = a rede nao distingue nada (SHA-256 parece aleatorio).")
    return results


def dump_csv(path, rounds, n, delta_bit, seed):
    delta = make_delta(bit=delta_bit)
    rng = np.random.default_rng(seed)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("rounds,label,m0_hex,m1_hex,c0_hex,c1_hex\n")
        for i in range(n):
            m0 = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
            c0 = sha256_block_rounds(m0, rounds)
            label = i & 1
            if label == 1:
                m1 = bytes(x ^ z for x, z in zip(m0, delta))
            else:
                m1 = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
            c1 = sha256_block_rounds(m1, rounds)
            fh.write(f"{rounds},{label},{m0.hex()},{m1.hex()},{c0.hex()},{c1.hex()}\n")
    print(f"Dataset de exemplo gravado em {path} ({n} linhas, {rounds} rondas).")


def main():
    ap = argparse.ArgumentParser(description="Distinguisher SHA-256 rondas reduzidas em CUDA")
    ap.add_argument('--rounds', type=int, nargs='+',
                    default=[2, 3, 4, 5, 6, 7, 8, 10, 16, 64],  # <--- LISTA CORRIGIDA AQUI
                    help="numeros de rondas a testar")
    ap.add_argument('--n-train', type=int, default=20000)
    ap.add_argument('--n-test', type=int, default=4000)
    ap.add_argument('--delta-bit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=456789) # fixo para reproductibilidade
    ap.add_argument('--dump-csv', type=int, metavar='RONDAS', default=None,
                    help="em vez de treinar, grava um CSV de exemplo")
    ap.add_argument('--csv-path', default='dataset_estudo4_exemplo.csv')
    ap.add_argument('--csv-n', type=int, default=10000)
    args = ap.parse_args()

    if args.dump_csv is not None:
        dump_csv(args.csv_path, args.dump_csv, args.csv_n, args.delta_bit, args.seed)
        return

    run_curve_cuda(args.rounds, args.n_train, args.n_test, args.delta_bit, args.seed)
    

if __name__ == '__main__':
    main()

