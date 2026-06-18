# 🎬 Análise de Sentimento IMDb

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat)
![ML](https://img.shields.io/badge/Scikit--Learn-PyTorch-orange?style=flat)
![UI](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat)

[![App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://maicc-sentiment-analysis-imdb.streamlit.app/)

## 🛠 Tecnologias
* **Linguagem:** Python 3.12
* **Bibliotecas ML:** Pandas, Scikit-Learn 1.8, NLTK, Matplotlib, Joblib
* **Interface:** Streamlit (Web App)

## 📂 Estrutura do Projeto
```text
sentiment-analysis-imdb/
├── data/           # Dataset (imdb-reviews-pt-br.csv)
├── models/         # Artefatos treinados (.pkl)
├── notebooks/      # Exploração e experimentos
├── src/            # Módulos de pré-processamento e lógica
├── app.py          # Interface Streamlit
└── requirements.txt
---

🚀 Como Rodar

1. Clonar e Ambiente
git clone [https://github.com/MaiccGms8/sentiment-analysis-imdb.git](https://github.com/MaiccGms8/sentiment-analysis-imdb.git)
cd sentiment-analysis-imdb
python -m venv .venv

# Ativação (Linux/Mac): 
source .venv/bin/activate

2. Instalação e Dataset
pip install -r requirements.txt

Coloque o dataset em data/imdb-reviews-pt-br.csv (disponível no Kaggle).

3. Execução
Treinar: python src/02_save_model.py

Interface Web: python -m streamlit run app.py

Terminal: python src/03_predict.py

🧠 Pipeline de NLP
Limpeza: Remoção de pontuação e stopwords (preservando negações).

Vetorização: TF-IDF com 5.000 features.

Classificação: Regressão Logística (Modelo Baseline) e testes com Deep Learning/BERT.

📊 Status
[x] Estrutura modular (preprocess.py)

[x] Modelo baseline (Regressão Logística)

[x] Interface Web (Streamlit)

[x] Verificação de versão (Scikit-Learn)

[x] Fine-tuning e Deep Learning (BERTimbau)

Desenvolvido por Maicon Gomes e Cezar Tosta