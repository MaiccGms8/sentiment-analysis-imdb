import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

_pt_stopwords = set(stopwords.words('portuguese')) - {'não', 'nem', 'nunca', 'nada', 'tampouco'}

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúç\s]', '', texto)
    palavras = texto.split()
    return " ".join(p for p in palavras if p not in _pt_stopwords)
