import pandas as pd
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Configuração inicial das Stopwords 
nltk.download('stopwords')

pt_stopwords = set(stopwords.words('portuguese')) - {'não', 'nem', 'nunca', 'nada', 'tampouco'}

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""

    # Normalização e Limpeza
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúç\s]', '', texto)

    # Tokenização e Filtragem 
    palavras = texto.split()
    palavras_limpas = [p for p in palavras if p not in pt_stopwords]
    
    return " ".join(palavras_limpas)

# Carregamento 
caminho_dados = os.path.join(os.path.dirname(__file__), '../data/imdb-reviews-pt-br.csv')
df = pd.read_csv(caminho_dados)

print("Limpando os textos (isso pode demorar um pouco)...")
df['texto_limpo'] = df['text_pt'].apply(limpar_texto)

# Vetorização e Modelo
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpo'])
y = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_final = LogisticRegression()
modelo_final.fit(X_train, y_train)

# Salvamento
if not os.path.exists('../models'):
    os.makedirs('../models')

joblib.dump(modelo_final, '../models/melhor_modelo_sentimento.pkl')
joblib.dump(vectorizer, '../models/vectorizer_tfidf.pkl')

print("Sucesso! Modelo e Vetorizador salvos.")