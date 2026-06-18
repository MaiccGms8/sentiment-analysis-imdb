import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import limpar_texto

# Define caminhos absolutos
base_dir = os.path.dirname(os.path.abspath(__file__))
caminho_dados = os.path.abspath(os.path.join(base_dir, '../data/imdb-reviews-pt-br.csv'))

print(" Carregando o dataset para o Tuning...")
df = pd.read_csv(caminho_dados)

if len(df) > 15000:
    print(" Reduzindo temporariamente para uma amostra de 15.000 linhas para otimizar o tempo de busca...")
    df = df.sample(n=15000, random_state=42).reset_index(drop=True)

print(" Limpando os textos da amostra...")
df['texto_limpo'] = df['text_pt'].apply(limpar_texto)

print(" Vetorizando com TF-IDF (5000 features)...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpo'])
y = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CONFIGURAÇÃO DO GRID SEARCH 
print("\n Configurando o Grid Search...")
modelo_base = LogisticRegression(max_iter=1000, random_state=42)

# Grade de parâmetros que queremos testar
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'saga']  
}

# Validação Cruzada de 3 dobras (3-Fold CV)
grid_search = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1 
)

print(" Iniciando a busca pelos melhores parâmetros (isso pode levar alguns minutos)...")
grid_search.fit(X_train, y_train)

# RESULTADOS 
print("\n --- RESULTADOS DO TUNING ---")
print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
print(f"Melhor acurácia na validação cruzada: {grid_search.best_score_:.4f}")

# Avaliando o melhor modelo 
melhor_modelo = grid_search.best_estimator_
predicoes = melhor_modelo.predict(X_test)

print("\n Relatório de Classificação do Melhor Modelo:")
print(classification_report(y_test, predicoes, target_names=['Negativo', 'Positivo']))