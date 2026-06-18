import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print(" Carregando o Tokenizador do BERTimbau (NeuralMind)...")
# Carrega o tokenizador oficial em português do Hugging Face
nome_modelo = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

print(" Carregando a arquitetura do Modelo...")
# Carrega o modelo preparado para classificação (2 classes: Positivo/Negativo)
model = AutoModelForSequenceClassification.from_pretrained(nome_modelo, num_labels=2)

# Frase de teste para ver o mecanismo de atenção em ação
frase = "Esse filme do IMDb não é tão ruim, eu achei muito bom!"

print(f"\n Frase original: '{frase}'")

# O tokenizador transforma o texto bruto em tensores numéricos para o PyTorch
tokens = tokenizer(frase, padding=True, truncation=True, return_tensors="pt")

print("\n --- TEXTO TRANSFORMADO PELO BERT ---")
print("Input IDs (Códigos numéricos de cada token):")
print(tokens['input_ids'])

print("\n Attention Mask (Diz ao modelo onde focar a atenção):")
print(tokens['attention_mask'])

# Convertendo os IDs de volta para palavras para ver como o BERT quebra o texto
palavras_convertidas = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
print("\n Como o BERT fatiou a frase internamente:")
print(palavras_convertidas)