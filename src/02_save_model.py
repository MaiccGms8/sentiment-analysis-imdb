import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import sklearn
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocess import limpar_texto

caminho_dados = os.path.join(os.path.dirname(__file__), '../data/imdb-reviews-pt-br.csv')
df = pd.read_csv(caminho_dados)

print("Limpando os textos (isso pode demorar um pouco)...")
df['texto_limpo'] = df['text_pt'].apply(limpar_texto)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpo'])
y = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_final = LogisticRegression()
modelo_final.fit(X_train, y_train)

caminho_models = os.path.join(os.path.dirname(__file__), '../models')
os.makedirs(caminho_models, exist_ok=True)

joblib.dump(modelo_final, os.path.join(caminho_models, 'melhor_modelo_sentimento.pkl'))
joblib.dump(vectorizer, os.path.join(caminho_models, 'vectorizer_tfidf.pkl'))
joblib.dump({'sklearn_version': sklearn.__version__}, os.path.join(caminho_models, 'metadata.pkl'))

print(f"Sucesso! Modelo e Vetorizador salvos (scikit-learn {sklearn.__version__}).")
