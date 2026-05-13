import joblib
import re
import nltk
import os
from nltk.corpus import stopwords

# Carregar as Stopwords 
nltk.download('stopwords', quiet=True)
pt_stopwords = set(stopwords.words('portuguese')) - {'não', 'nem', 'nunca', 'nada', 'tampouco'}

# 2. A função de limpeza 
def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúç\s]', '', texto)
    palavras = texto.split()
    palavras_limpas = [p for p in palavras if p not in pt_stopwords]
    return " ".join(palavras_limpas)

# 3. Carregar os artefatos (.pkl)
base_path = os.path.dirname(__file__)
modelo = joblib.load(os.path.join(base_path, '../models/melhor_modelo_sentimento.pkl'))
vectorizer = joblib.load(os.path.join(base_path, '../models/vectorizer_tfidf.pkl'))

def classificar_frase(texto):
    texto_limpo = limpar_texto(texto) 
    vetor = vectorizer.transform([texto_limpo])
    predicao = modelo.predict(vetor)
    
    probabilidade = modelo.predict_proba(vetor)[0]
    
    sentimento = "Positivo" if predicao[0] == 1 else "Negativo"
    return sentimento

# 4. Teste manual
frase = "Esse filme foi uma perda de tempo total, odiei."
resultado = classificar_frase(frase)
print(f"\nFrase: {frase}")
print(f"Resultado: {resultado}")