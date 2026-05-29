import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
from preprocess import limpar_texto

base_path = os.path.dirname(__file__)
modelo = joblib.load(os.path.join(base_path, '../models/melhor_modelo_sentimento.pkl'))
vectorizer = joblib.load(os.path.join(base_path, '../models/vectorizer_tfidf.pkl'))

def classificar_frase(texto):
    texto_limpo = limpar_texto(texto)
    vetor = vectorizer.transform([texto_limpo])
    predicao = modelo.predict(vetor)[0]
    probabilidade = modelo.predict_proba(vetor)[0]
    sentimento = "Positivo" if predicao == 1 else "Negativo"
    confianca = probabilidade[predicao] * 100
    return sentimento, confianca

if __name__ == '__main__':
    frase = "Esse filme foi uma perda de tempo total, odiei."
    sentimento, confianca = classificar_frase(frase)
    print(f"\nFrase: {frase}")
    print(f"Resultado: {sentimento} ({confianca:.1f}% de confiança)")
