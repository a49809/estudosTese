#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo - Inversao do SHA-256 por rede neuronal (Aceleração CUDA)
==================================================================
"""

import argparse
import hashlib
import sys
import numpy as np

# Substituição do scikit-learn pelo PyTorch para suporte nativo a CUDA
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def bits_of_batch(b_array):
    """
    Converte uma matriz de bytes (NumPy uint8) num tensor PyTorch de bits (0/1).
    Processamento vetorizado para evitar loops pesados em CPU.
    """
    # np.unpackbits expande os bytes para bits ao nível do lote (axis=1)
    unpacked = np.unpackbits(b_array, axis=1)
    return torch.from_numpy(unpacked).float()


def make_pairs_cuda(n, seed, device):
    """
    Gera n pares (hash_bits -> key_bits) utilizando CPU para a criptografia (SHA-256)
    e envia imediatamente os lotes de bits expandidos para a memória da GPU (CUDA).
    """
    # Gerador pseudoaleatório baseado na semente fornecida
    rng = np.random.default_rng(seed)
    
    # Gerar todas as chaves em lote (32 bytes por chave)
    keys_raw = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    hashes_raw = np.zeros((n, 32), dtype=np.uint8)
    
    # O cálculo do SHA-256 individual é feito via CPU devido às restrições da biblioteca hashlib
    for i in range(n):
        hashes_raw[i] = np.frombuffer(hashlib.sha256(keys_raw[i].tobytes()).digest(), dtype=np.uint8)
    
    # Conversão vetorizada para matrizes de bits e migração direta para CUDA
    X = bits_of_batch(hashes_raw).to(device)
    Y = bits_of_batch(keys_raw).to(device)
    
    return X, Y


def mean_bits_correct_cuda(y_true, y_pred):
    """
    Avaliação matemática executada inteiramente dentro dos núcleos CUDA da GPU.
    Retorna a média de bits corretos por amostra.
    """
    # Binariza as previsões com threshold de 0.5
    preds_binary = (y_pred > 0.5).float()
    
    # .sum(dim=1) conta os acertos (inteiros)
    # .float() converte os inteiros para float para permitir o cálculo da média
    return float((y_true == preds_binary).sum(dim=1).float().mean().item())


# Definição de uma Rede Neuronal equivalente ao MLPClassifier em PyTorch
class MLP_SHA256(nn.Module):
    def __init__(self):
        super(MLP_SHA256, self).__init__()
        self.network = nn.Sequential(
            
            nn.Linear(256, 384), #estrutura de rede: 256,384,256,128,256,384,256
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            
            nn.Sigmoid() # Camada de saída a mapear probabilidades entre 0 e 1
        )
        
    def forward(self, x):
        return self.network(x)


def run(n_train, n_test, seed):
    if not HAVE_TORCH:
        sys.exit("PyTorch não encontrado. Instale previamente: pip install torch")

    # Define o dispositivo: prioriza GPU via CUDA, caso contrário faz fallback para CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo ativo para computação: {device.type.upper()}")
    if device.type == "cpu":
        print("[!] INFO: GPU CUDA não detetada. A correr em modo de compatibilidade (CPU).")

    # Fixar sementes do PyTorch para reprodutibilidade
    torch.manual_seed(seed)

    print("[*] A produzir amostras e a transferir para a GPU...")
    Xtr, Ytr = make_pairs_cuda(n_train, seed, device)
    Xte, Yte = make_pairs_cuda(n_test, seed + 987654321, device)

    print("=" * 66)
    print("Estudo 3 refeito - Inversao do SHA-256 com CUDA / PyTorch")
    print("=" * 66)
    print(f"Treino = {n_train} chaves | Teste = {n_test} chaves (nunca vistas)")
    print("-" * 66)

    # Inicializa o modelo no dispositivo selecionado (GPU/CUDA)
    model = MLP_SHA256().to(device)
    
    # Equivalente ao solver='adam' e alpha=1e-5 (weight_decay) do sklearn
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss() # Binary Cross Entropy para saídas multi-label independentes

    print("A treinar a rede em CUDA...")
    model.train()
    
    # Sistema de mini-lotes (batch_size=256) replicando o comportamento original
    batch_size = 256
    epochs = 10000 #outros limites usados em testes: 120, 240, 360
    
    for epoch in range(epochs):
        permutation = torch.randperm(Xtr.size(0))
        for i in range(0, Xtr.size(0), batch_size):
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = Xtr[indices], Ytr[indices]
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()

    # Fase de Inferência e Avaliação 
    model.eval()
    with torch.no_grad():
        pred_tr = model(Xtr)
        pred_te = model(Xte)

        # Avaliação executada na GPU
        bt = mean_bits_correct_cuda(Ytr, pred_tr)
        be = mean_bits_correct_cuda(Yte, pred_te)

    print("-" * 66)
    print(f"{'':28} {'bits acertados / 256':>22} {'%':>8}")
    print(f"{'TREINO (amostras vistas)':28} {bt:>22.1f} {bt/256*100:>7.1f}%")
    print(f"{'TESTE  (amostras novas)':28} {be:>22.1f} {be/256*100:>7.1f}%")
    print(f"{'Baseline do acaso':28} {128.0:>22.1f} {50.0:>7.1f}%")
    print("-" * 66)

    # Teste estatístico Z-score (Calculado localmente na GPU)
    z = (be - 128.0) / (8.0 / np.sqrt(n_test))
    print(f"Distancia do teste ao acaso: {be - 128.0:+.1f} bits  (z = {z:+.1f})")
    if abs(z) < 3:
        print("=> O TESTE esta ao nivel do acaso: a rede NAO inverte o SHA-256.")
    else:
        print("=> Sinal acima do acaso no teste: investigar (nao esperado a 64 rondas).")
    
    gap = bt - be
    print(f"Gap treino-teste: {gap:+.1f} bits  ->  mede a MEMORIZACAO.")
    print("=" * 66)


def main():

    run(10000, 2500, 456789) #n-treino, n-teste, seed (fixo para reprodutibilidade)


if __name__ == '__main__':
    main()
