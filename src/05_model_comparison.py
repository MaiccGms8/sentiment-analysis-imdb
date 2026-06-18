import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import limpar_texto

base_dir = os.path.dirname(os.path.abspath(__file__))
caminho_dados = os.path.abspath(os.path.join(base_dir, '../data/imdb-reviews-pt-br.csv'))

print(" Carregando o dataset para a Comparação...")
df = pd.read_csv(caminho_dados)

if len(df) > 10000:
    print(" Utilizando uma amostra de 10.000 linhas para viabilizar o tempo de treino da Random Forest...")
    df = df.sample(n=10000, random_state=42).reset_index(drop=True)

print(" Limpando os textos da amostra...")
df['texto_limpo'] = df['text_pt'].apply(limpar_texto)

print(" Vetorizando com TF-IDF (5000 features)...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpo'])
y = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

# Divisão estável de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dicionário contendo os competidores
modelos = {
    "Regressão Logística": LogisticRegression(C=1.0, solver='saga', max_iter=1000, random_state=42),
    "Naive Bayes (Multinomial)": MultinomialNB(),
    "Random Forest (100 árvores)": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    # Limitamos o max_depth da árvore para o treino ser rápido e não estourar a memória RAM
}

resultados_acuracia = {}

print("\n Iniciando os combates na Arena de Modelos...")
print("-" * 50)

for nome, modelo in modelos.items():
    print(f"Treinando o modelo: {nome}...")
    modelo.fit(X_train, y_train)
    
    print(f"Avaliando {nome} nos dados de teste...")
    predicoes = modelo.predict(X_test)
    
    acuracia = accuracy_score(y_test, predicoes)
    resultados_acuracia[nome] = acuracia
    
    print(f"-> Acurácia do {nome}: {acuracia:.4f}")
    print(f"\n Relatório de Classificação para {nome}:")
    print(classification_report(y_test, predicoes, target_names=['Negativo', 'Positivo']))
    print("-" * 50)

# RESUMO FINAL
print("\n --- PLACAR FINAL DA ARENA ---")
df_resultados = pd.DataFrame(list(resultados_acuracia.items()), columns=['Modelo', 'Acurácia']).sort_values(by='Acurácia', ascending=False)
print(df_resultados.to_string(index=False))

print("\n Sugestão: Olhe os relatórios acima e veja quem teve o melhor F1-Score!")