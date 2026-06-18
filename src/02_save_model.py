import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sklearn
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocess import limpar_texto

base_dir = os.path.dirname(os.path.abspath(__file__))
caminho_dados = os.path.abspath(os.path.join(base_dir, '../data/imdb-reviews-pt-br.csv'))
caminho_models = os.path.abspath(os.path.join(base_dir, '../models'))

print("Carregando o dataset...")
df = pd.read_csv(caminho_dados)

print("Limpando os textos (isso pode demorar cerca de 1 a 2 minutos)...")
df['texto_limpo'] = df['text_pt'].apply(limpar_texto)

print("Vetorizando os dados com TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpo'])
y = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Treinando o modelo de Regressão Logística...")
modelo_final = LogisticRegression(max_iter=1000)
modelo_final.fit(X_train, y_train)

os.makedirs(caminho_models, exist_ok=True)

print("Salvando os artefatos (.pkl) na pasta models/...")
joblib.dump(modelo_final, os.path.join(caminho_models, 'melhor_modelo_sentimento.pkl'))
joblib.dump(vectorizer, os.path.join(caminho_models, 'vectorizer_tfidf.pkl'))
joblib.dump({'sklearn_version': sklearn.__version__}, os.path.join(caminho_models, 'metadata.pkl'))

print(f"\n Sucesso absoluto! Tudo salvo corretamente (scikit-learn {sklearn.__version__}).")