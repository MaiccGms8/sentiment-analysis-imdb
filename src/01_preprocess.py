import re
import nltk
from nltk.corpus import stopwords

# Baixa as palavras vazias 
nltk.download('stopwords')

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""

    # Deixar tudo em minúsculo
    texto = texto.lower()

    # Remover pontuação e números (mantendo acentos)
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúç\s]', '', texto)

    # Tratar Stopwords
    pt_stopwords = set(stopwords.words('portuguese'))
    
    # Observação: Não podemos remover palavras de negação em análise de sentimento
    palavras_negacao = {'não', 'nem', 'nunca', 'nada', 'tampouco'}
    pt_stopwords = pt_stopwords - palavras_negacao
    
    # Tokenização (separar palavras e limpeza)
    palavras = texto.split()
    palavras_limpas = [p for p in palavras if p not in pt_stopwords]
    
    return " ".join(palavras_limpas)