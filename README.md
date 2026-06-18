## Análise de Sentimento - IMDb PT-BR

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard_BI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Data_Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://maicc-sentiment-analysis-imdb.streamlit.app/](https://github.com/MaiccGms8/sentiment-analysis-imdb))

Classificação de avaliações de filmes em português como **positivas** ou **negativas** usando NLP e Machine Learning, com interface web interativa via Streamlit.

**Dataset:** [IMDb Reviews PT-BR no Kaggle](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr)

---

## Tecnologias

- Python 3.12
- Pandas, Scikit-Learn 1.8, NLTK, Matplotlib
- Streamlit (interface web)
- Joblib (serialização de modelos)

---

## Estrutura do projeto

```
sentiment-analysis-imdb/
├── data/                          # Dataset CSV (não versionado)
│   └── imdb-reviews-pt-br.csv
├── models/                        # Artefatos gerados pelo treinamento
│   ├── melhor_modelo_sentimento.pkl
│   ├── vectorizer_tfidf.pkl
│   └── metadata.pkl               # Versão do scikit-learn usada no treino
├── notebooks/
│   ├── 01_exploracao_e_modelagem.ipynb
│   └── 02_experimentacao.ipynb
├── src/
│   ├── preprocess.py              # Limpeza de texto (módulo compartilhado)
│   ├── 01_preprocess.py           # Demonstração da limpeza
│   ├── 02_save_model.py           # Treinamento e salvamento do modelo
│   └── 03_predict.py              # Predição via terminal
├── app.py                         # Interface Streamlit
└── requirements.txt
```

---

## Como rodar

### 1. Clonar o repositório

```bash
git clone https://github.com/MaiccGms8/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv

# Git Bash / Linux / Mac:
source .venv/Scripts/activate

# PowerShell:
.venv\Scripts\Activate.ps1

# CMD:
.venv\Scripts\activate.bat
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar o dataset

Acesse o Kaggle e faça login:
[https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr)

Baixe o arquivo `imdb-reviews-pt-br.csv` e coloque-o em:

```
sentiment-analysis-imdb/
└── data/
    └── imdb-reviews-pt-br.csv
```

> A pasta `data/` não é versionada no git — crie-a manualmente antes de copiar o arquivo.

### 5. Treinar o modelo

```bash
python src/02_save_model.py
```

Gera três arquivos em `models/`:
- `melhor_modelo_sentimento.pkl` — modelo de Regressão Logística treinado
- `vectorizer_tfidf.pkl` — vetorizador TF-IDF
- `metadata.pkl` — versão do scikit-learn usada no treino

> Sempre execute este script com o mesmo ambiente Python usado para rodar o app. Se houver incompatibilidade de versão do scikit-learn, o app bloqueará a execução com uma mensagem de erro.

### 6. Iniciar a interface web

```bash
python -m streamlit run app.py
```

Acesse `http://localhost:8501` no navegador, digite uma avaliação de filme em português e clique em **Classificar**.

> Use `python -m streamlit` em vez de `streamlit` diretamente para garantir que o executável do `.venv` seja usado, evitando conflitos com outros ambientes Python (ex: Anaconda) instalados no sistema.

### 7. (Opcional) Testar via terminal

```bash
python src/03_predict.py
```

Saída esperada:
```
Frase: Esse filme foi uma perda de tempo total, odiei.
Resultado: Negativo (94.3% de confiança)
```

### 8. (Opcional) Explorar os notebooks

```bash
jupyter lab
```

---

## Pipeline de NLP

1. **Limpeza** (`preprocess.py`) — lowercase, remoção de pontuação e stopwords em português (preservando negações: *não, nem, nunca, nada, tampouco*)
2. **Vetorização** — TF-IDF com 5.000 features
3. **Classificação** — Regressão Logística binária (positivo / negativo)

---

## Status

- [x] Estrutura do repositório configurada
- [x] Limpeza de texto modularizada (`preprocess.py`)
- [x] Modelo baseline (Regressão Logística) treinado e salvo
- [x] Interface web com Streamlit
- [x] Verificação de compatibilidade de versão do scikit-learn
- [x] Comparação entre modelos (Naive Bayes, Random Forest)
- [x] Tuning de hiperparâmetros
- [x] Modelos de Deep Learning / BERT em português

---

**Desenvolvido por Maicon Gomes e Cezar Tosta**
