import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('portuguese')

def limpar_texto(texto):
    # Remove pontuação e caracteres especiais
    texto = re.sub(r'[^\w\s]', '', texto)
    # Tudo para minúsculo
    texto = texto.lower()
    # Remove stopwords
    texto = " ".join([palavra for palavra in texto.split() if palavra not in stop_words])
    return texto