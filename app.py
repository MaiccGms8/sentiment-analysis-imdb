import streamlit as st
import joblib
import os
import sys
import pandas as pd
import plotly.express as px

# Garante que o Python encontre o script preprocess na pasta src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import limpar_texto

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Análise de Sentimento",
    page_icon="📊",
    layout="wide"
)

# --- CARREGAR MODELOS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
caminho_modelo = os.path.join(base_dir, 'models/melhor_modelo_sentimento.pkl')
caminho_vectorizer = os.path.join(base_dir, 'models/vectorizer_tfidf.pkl')

@st.cache_resource
def carregar_artefatos():
    model = joblib.load(caminho_modelo)
    vectorizer = joblib.load(caminho_vectorizer)
    return model, vectorizer

try:
    model, vectorizer = carregar_artefatos()
except Exception as e:
    st.error(f"Erro ao carregar os models. Certifique-se de que rodou o script 02. Erro: {e}")
    st.stop()

# --- GERENCIAMENTO DE ESTADO (HISTÓRICO) ---
if 'historico' not in st.session_state:
    st.session_state.historico = []

# --- INTERFACE VISUAL ---
st.title(" Painel Analítico de Sentimentos — IMDb")
st.markdown("Insira a crítica de um filme em português para analisar a polaridade do texto com Machine Learning clássico calibrado.")

# Criando duas colunas na tela principal
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.subheader(" Entrada de Dados")
    frase_usuario = st.text_area(
        "Digite ou cole a crítica do filme aqui:",
        placeholder="Ex: O filme tem uma fotografia espetacular, mas o roteiro perde o ritmo no segundo ato...",
        height=150
    )
    
    botao_analisar = st.button("Executar Análise de Dados", type="primary")

with col2:
    st.subheader(" Resultado do Modelo")
    
    if botao_analisar and frase_usuario.strip() != "":
        # 1. Pipeline de NLP em tempo real (Restaurado)
        texto_limpo = limpar_texto(frase_usuario)
        vetor_texto = vectorizer.transform([texto_limpo])
        
        # 2. Predição baseada diretamente nas probabilidades (Alinhado contra inversão)
        probabilidades = model.predict_proba(vetor_texto)[0] # [Prob_Neg, Prob_Pos]
        prob_neg = probabilidades[0]
        prob_pos = probabilidades[1]
        
        # Define o resultado com base na maior probabilidade
        if prob_pos >= prob_neg:
            classe_resultado = "Positivo"
            confianca = prob_pos
        else:
            classe_resultado = "Negativo"
            confianca = prob_neg
        
        # 3. Guardar no histórico da sessão
        st.session_state.historico.insert(0, {"Texto": frase_usuario[:60] + "...", "Sentimento": classe_resultado})

        # 4. Exibição de Métricas Visuais Dinâmicas
        if classe_resultado == "Positivo":
            st.success(f"### Sentimento Detectado: {classe_resultado} ")
            st.metric(label="Grau de Certeza do Algoritmo", value=f"{confianca*100:.2f}%", delta="Classificação Confiável")
        else:
            st.error(f"### Sentimento Detectado: {classe_resultado} ")
            st.metric(label="Grau de Certeza do Algoritmo", value=f"{confianca*100:.2f}%", delta="- Classificação Confiável", delta_color="inverse")
            
        # 5. Estruturação correta do DataFrame para o Plotly
        df_proba = pd.DataFrame({
            'Sentimento': ['Negativo', 'Positivo'],
            'Probabilidade': [prob_neg * 100, prob_pos * 100]
        })
        
        fig = px.bar(
            df_proba, 
            x='Probabilidade', 
            y='Sentimento', 
            orientation='h',
            text='Probabilidade',
            color='Sentimento',
            color_discrete_map={'Positivo': '#2ecc71', 'Negativo': '#e74c3c'},
            range_x=[0, 100]
        )
        
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            height=200, 
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis_title="Confiança (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Aguardando a inserção de texto para processar os gráficos e indicadores.")

# --- SEÇÃO INFERIOR: HISTÓRICO DE INTERAÇÕES (STORYTELLING/BI) ---
st.write("---")
st.subheader(" Histórico de Consultas da Sessão")

if st.session_state.historico:
    df_hist = pd.DataFrame(st.session_state.historico)
    st.dataframe(df_hist, use_container_width=True)
else:
    st.caption("Nenhuma crítica analisada nesta sessão ainda.")