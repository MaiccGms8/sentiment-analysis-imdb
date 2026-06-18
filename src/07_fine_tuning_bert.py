import sys
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Garante que o Python encontre módulos internos se necessário
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import limpar_texto

# 1. Configuração de Caminhos e Dispositivo (GPU ou CPU)
base_dir = os.path.dirname(os.path.abspath(__file__))
caminho_dados = os.path.abspath(os.path.join(base_dir, '../data/imdb-reviews-pt-br.csv'))
caminho_modelo_salvo = os.path.abspath(os.path.join(base_dir, '../models/bertimbau_sentimento'))

# Verifica se o seu PC tem uma GPU dedicada utilizável (CUDA). Se não, usa a CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Rodando o treinamento utilizando: {device.upper()}")

# 2. Carregando e Amostrando os Dados
print(" Carregando o dataset para o Fine-Tuning...")
df = pd.read_csv(caminho_dados)

# BERTimbau é massivo. Para o treino na CPU rodar rápido como teste,
# usaremos uma amostra bem controlada de 50 linhas.
if len(df) > 50:
    print(" Reduzindo para uma amostra de teste de 50 linhas para rodar rápido na CPU...")
    df = df.sample(n=50, random_state=42).reset_index(drop=True)

print(" Aplicando limpeza leve nos textos...")
df['texto_limpo'] = df['text_pt'].apply(limpar_texto)

# 3. Tokenização para o BERTimbau
print(" Carregando o Tokenizador do BERTimbau...")
nome_modelo = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

# Mapeia as labels para números (0 para negativo, 1 para positivo)
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

# Divisão de Treino e Validação (Uso de .tolist() para blindar contra o PyArrow)
X_train, X_val, y_train, y_val = train_test_split(
    df['texto_limpo'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

print(" Convertendo textos em Tensores do BERT (Tokenização)...")
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=256)

# 4. Criando a estrutura de Dataset do PyTorch que o Trainer exige
class DatasetIMDb(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])