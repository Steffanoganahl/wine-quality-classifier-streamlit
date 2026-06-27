# 🍷 Wine Quality Classifier — Streamlit

App interactiva para clasificación de calidad de vino tinto usando K-Nearest Neighbors (KNN), desplegada con Streamlit.

**Accuracy del modelo: 84%** — entrenado con Red Wine Quality Dataset (UCI, 1.599 muestras).

---

## Estructura del proyecto

```
wine-streamlit/
├── streamlit_app.py
├── requirements_streamlit.txt
└── model/
    ├── knn_wine_model.pkl
    └── scaler.pkl
```

---

## Instalación local

```bash
# 1. Crear entorno virtual
/usr/local/opt/python@3.12/bin/python3.12 -m venv venv

# 2. Activar entorno
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements_streamlit.txt

# 4. Correr la app
streamlit run streamlit_app.py
```

Abrir en: [http://localhost:8501](http://localhost:8501)

---

## Despliegue en Render

1. Subir este repositorio a GitHub (incluyendo la carpeta `model/`)
2. En [render.com](https://render.com): **New → Web Service → conectar repo**
3. Configurar:
   - **Build Command:** `pip install -r requirements_streamlit.txt`
   - **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan:** Free
4. Click **Deploy Web Service**

---

## Funcionamiento

La app permite ajustar los 11 parámetros fisicoquímicos del vino mediante **sliders interactivos** y muestra en tiempo real:
- Clasificación de calidad: **Baja / Media / Alta**
- Probabilidad por cada clase
- Comparación de los valores ingresados con los promedios históricos de cada clase
- Estadísticas del modelo: métricas, relevancia de features y distribución de clases

## Caso de uso

Este clasificador puede ser útil para bodegas y enólogos que quieran evaluar el perfil fisicoquímico de un lote antes de embotellarlo, orientando decisiones de corrección o descarte sin depender exclusivamente de cata sensorial.

## Features del modelo

| Feature | Descripción |
|---|---|
| fixed acidity | Acidez fija — tartárico (g/L) |
| volatile acidity | Acidez volátil — acético (g/L) |
| citric acid | Ácido cítrico (g/L) |
| residual sugar | Azúcar residual (g/L) |
| chlorides | Cloruros — sal (g/L) |
| free sulfur dioxide | SO₂ libre (mg/L) |
| total sulfur dioxide | SO₂ total (mg/L) |
| density | Densidad (g/mL) |
| pH | pH |
| sulphates | Sulfatos (g/L) |
| alcohol | Alcohol (% vol.) |

## Clases predichas

| Clase | Etiqueta | Quality original |
|---|---|---|
| 0 | Baja 🍷 | ≤ 4 |
| 1 | Media 🍷🍷 | 5 – 6 |
| 2 | Alta 🍷🍷🍷 | ≥ 7 |

---

## Tech stack

- Python 3.12
- scikit-learn (KNN, GridSearchCV, StandardScaler)
- Streamlit 1.x
- joblib

---

## Render Live Website
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render)](https://wine-quality-classifier-streamlit.onrender.com/)

*Steffano · 4Geeks Academy DS & ML Bootcamp*
