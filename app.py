import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import sklearn
import streamlit as st
import joblib
from preprocess import limpar_texto

def _verificar_versao():
    meta_path = os.path.join(os.path.dirname(__file__), 'models/metadata.pkl')
    if not os.path.exists(meta_path):
        return
    meta = joblib.load(meta_path)
    saved_version = meta.get('sklearn_version', 'desconhecida')
    if saved_version != sklearn.__version__:
        st.error(
            f"**Incompatibilidade de versão do scikit-learn**\n\n"
            f"- Modelo treinado com: `{saved_version}`\n"
            f"- Versão atual: `{sklearn.__version__}`\n\n"
            "Execute o comando abaixo **no mesmo ambiente** em que o app está rodando e reinicie:\n"
            "```\npython src/02_save_model.py\n```"
        )
        st.stop()

@st.cache_resource
def carregar_modelo():
    base = os.path.dirname(__file__)
    modelo = joblib.load(os.path.join(base, 'models/melhor_modelo_sentimento.pkl'))
    vectorizer = joblib.load(os.path.join(base, 'models/vectorizer_tfidf.pkl'))
    return modelo, vectorizer

def classificar(texto, modelo, vectorizer):
    texto_limpo = limpar_texto(texto)
    vetor = vectorizer.transform([texto_limpo])
    predicao = modelo.predict(vetor)[0]
    probabilidade = modelo.predict_proba(vetor)[0]
    sentimento = "Positivo" if predicao == 1 else "Negativo"
    confianca = probabilidade[predicao] * 100
    return sentimento, confianca

# --- Layout ---
st.set_page_config(page_title="Análise de Sentimento", page_icon="🎬")
st.title("🎬 Análise de Sentimento - IMDb PT-BR")
st.caption("Digite uma avaliação de filme em português e descubra se o sentimento é positivo ou negativo.")

_verificar_versao()
modelo, vectorizer = carregar_modelo()

frase = st.text_area("Avaliação:", placeholder="Ex: Esse filme foi incrível, uma obra-prima!", height=120)

if st.button("Classificar", type="primary"):
    if not frase.strip():
        st.warning("Digite uma avaliação antes de classificar.")
    else:
        sentimento, confianca = classificar(frase, modelo, vectorizer)

        if sentimento == "Positivo":
            st.success(f"Sentimento: **{sentimento}**")
        else:
            st.error(f"Sentimento: **{sentimento}**")

        st.progress(int(confianca))
        st.caption(f"Confiança do modelo: {confianca:.1f}%")
