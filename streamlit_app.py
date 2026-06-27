"""
streamlit_app.py
----------------
Estructura de carpetas requerida:
    proyecto/
    ├── streamlit_app.py
    └── model/
        ├── knn_wine_model.pkl
        └── scaler.pkl

Instalar dependencias:
    pip install streamlit joblib numpy pandas scikit-learn

Ejecutar:
    streamlit run streamlit_app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- Configuración de página ---
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CSS personalizado ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');

    .main { background-color: #0f0a0a; }

    .wine-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #c9a84c;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .wine-sub {
        color: #9e8888;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .section-title {
        color: #c9a84c;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.5rem 0;
    }
    .badge {
        background: #8b1a1a;
        color: white;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* Slider track fondo */
    .stSlider [data-baseweb="slider"] > div:first-child {
        background: #2e1e1e !important;
    }
    /* Parte activa del slider */
    .stSlider [data-baseweb="slider"] [role="progressbar"] {
        background: #8b1a1a !important;
    }
    /* Bolita del slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #c9a84c !important;
        border: 2px solid #c9a84c !important;
        box-shadow: 0 0 6px #c9a84c88 !important;
        width: 18px !important;
        height: 18px !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        background: #e2c06a !important;
        box-shadow: 0 0 10px #c9a84cbb !important;
    }
    /* Label del slider */
    .stSlider label {
        color: #9e8888 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Cargar modelo ---
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "model", "knn_wine_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ No se pudo cargar el modelo: {e}\n\nAsegúrate de haber ejecutado `save_model.py` primero.")
    model_loaded = False
    st.stop()


# --- Constantes ---
FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

FEATURE_RANGES = {
    "fixed acidity":        (4.6,  15.9, 7.4,  0.1),
    "volatile acidity":     (0.12, 1.58, 0.70, 0.01),
    "citric acid":          (0.0,  1.0,  0.0,  0.01),
    "residual sugar":       (0.9,  15.5, 1.9,  0.1),
    "chlorides":            (0.012,0.611,0.076,0.001),
    "free sulfur dioxide":  (1.0,  72.0, 11.0, 1.0),
    "total sulfur dioxide": (6.0,  289.0,34.0, 1.0),
    "density":              (0.990,1.004,0.9978,0.0001),
    "pH":                   (2.74, 4.01, 3.51, 0.01),
    "sulphates":            (0.33, 2.0,  0.56, 0.01),
    "alcohol":              (8.4,  14.9, 9.4,  0.1),
}

QUALITY_CONFIG = {
    0: {"label": "Baja",  "emoji": "🍷",      "color": "#c0392b", "bg": "#2d0a0a"},
    1: {"label": "Media", "emoji": "🍷🍷",    "color": "#e67e22", "bg": "#2d1a00"},
    2: {"label": "Alta",  "emoji": "🍷🍷🍷",  "color": "#27ae60", "bg": "#0a2d15"},
}


# --- Header ---
st.markdown('<div class="wine-title">🍷 Wine Quality Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="wine-sub">Predicción de calidad de vino tinto mediante propiedades fisicoquímicas</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center"><span class="badge">KNN · Accuracy 84% · 4Geeks Academy</span></div>', unsafe_allow_html=True)

# --- Caso de uso real (punto 5) ---
st.markdown("""
<div style="background:#1a1010; border-left:3px solid #c9a84c; border-radius:0 8px 8px 0;
            padding:0.9rem 1.2rem; margin:0.5rem 0 1.2rem 0; color:#9e8888; font-size:0.88rem;">
    🏭 <strong style="color:#c9a84c;">Caso de uso</strong> — Este clasificador puede ser útil para
    <strong style="color:#e8d8d8;">bodegas y enólogos</strong> que quieran evaluar el perfil fisicoquímico
    de un lote <strong style="color:#e8d8d8;">antes de embotellarlo</strong>, orientando decisiones de
    corrección o descarte sin depender exclusivamente de cata sensorial.
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Sección de estadísticas del modelo ---
with st.expander("📈 Estadísticas del modelo", expanded=True):

    st.markdown('<div class="section-title">Métricas de rendimiento</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  "84%")
    m2.metric("Precision", "80%", help="Weighted avg")
    m3.metric("Recall",    "84%", help="Weighted avg")
    m4.metric("F1-Score",  "81%", help="Weighted avg")

    st.markdown("""
    <div style="color:#9e8888; font-size:0.78rem; margin-top:0.4rem;">
    Evaluado sobre 320 muestras de test · GridSearchCV con RepeatedStratifiedKFold (5 splits × 3 repeats)
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col_feat, col_dist = st.columns(2)

    with col_feat:
        st.markdown('<div class="section-title">Relevancia de features</div>', unsafe_allow_html=True)
        st.caption("Correlación absoluta con la calidad del vino")

        feature_corr = {
            "alcohol":              0.476,
            "volatile acidity":     0.391,
            "sulphates":            0.251,
            "citric acid":          0.226,
            "total sulfur dioxide": 0.185,
            "density":              0.175,
            "chlorides":            0.129,
            "fixed acidity":        0.124,
            "pH":                   0.058,
            "free sulfur dioxide":  0.051,
            "residual sugar":       0.014,
        }

        corr_df = pd.DataFrame(
            list(feature_corr.items()),
            columns=["Feature", "Correlación"]
        ).sort_values("Correlación", ascending=False)

        for _, row in corr_df.iterrows():
            bar_w = int(row["Correlación"] * 200)
            st.markdown(f"""
            <div style="margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#9e8888;">
                    <span>{row["Feature"]}</span>
                    <span style="color:#c9a84c;">{row["Correlación"]:.3f}</span>
                </div>
                <div style="background:#2e1e1e; border-radius:3px; height:5px; margin-top:2px;">
                    <div style="background:#8b1a1a; width:{bar_w}px; height:5px; border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_dist:
        st.markdown('<div class="section-title">Distribución de clases</div>', unsafe_allow_html=True)
        st.caption("Red Wine Quality Dataset · 1,599 muestras")

        totals = [63, 1319, 217]
        labels = ["Baja 🍷 (≤4)", "Media 🍷🍷 (5-6)", "Alta 🍷🍷🍷 (≥7)"]
        pcts   = ["3.9%", "82.5%", "13.6%"]
        colors = ["#c0392b", "#e67e22", "#27ae60"]
        total  = sum(totals)

        for i in range(3):
            bar_w = int(totals[i] / total * 180)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#9e8888;">
                    <span>{labels[i]}</span>
                    <span style="color:{colors[i]};">{totals[i]} · {pcts[i]}</span>
                </div>
                <div style="background:#2e1e1e; border-radius:3px; height:8px; margin-top:3px;">
                    <div style="background:{colors[i]}; width:{bar_w}px; height:8px; border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color:#9e8888; font-size:0.75rem; margin-top:0.8rem; padding-top:0.6rem; border-top:1px solid #2e1e1e;">
        ⚠️ Dataset desbalanceado — la clase Media concentra el 82.5% de los datos,
        lo que explica el recall más alto en esa categoría.
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- Inputs ---
st.markdown('<div class="section-title">Propiedades Fisicoquímicas</div>', unsafe_allow_html=True)

FEATURE_DESCRIPTIONS = {
    "fixed acidity":        "Acidez fija — tartárico (g/L)",
    "volatile acidity":     "Acidez volátil — acético (g/L)",
    "citric acid":          "Ácido cítrico (g/L)",
    "residual sugar":       "Azúcar residual (g/L)",
    "chlorides":            "Cloruros — sal (g/L)",
    "free sulfur dioxide":  "SO₂ libre (mg/L)",
    "total sulfur dioxide": "SO₂ total (mg/L)",
    "density":              "Densidad (g/mL)",
    "pH":                   "pH",
    "sulphates":            "Sulfatos (g/L)",
    "alcohol":              "Alcohol (% vol.)",
}

values = {}

col1, col2 = st.columns(2)

features_col1 = FEATURE_NAMES[:6]
features_col2 = FEATURE_NAMES[6:]

with col1:
    for feat in features_col1:
        mn, mx, default, step = FEATURE_RANGES[feat]
        values[feat] = st.slider(
            FEATURE_DESCRIPTIONS[feat],
            min_value=float(mn),
            max_value=float(mx),
            value=float(default),
            step=float(step),
        )

with col2:
    for feat in features_col2:
        mn, mx, default, step = FEATURE_RANGES[feat]
        values[feat] = st.slider(
            FEATURE_DESCRIPTIONS[feat],
            min_value=float(mn),
            max_value=float(mx),
            value=float(default),
            step=float(step),
        )

st.divider()

# --- Predicción automática en tiempo real (sin botón) ---
st.markdown('<div class="section-title">Resultado en tiempo real</div>', unsafe_allow_html=True)

input_values = [values[f] for f in FEATURE_NAMES]
input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

try:
    scaled = scaler.transform(input_df)
    prediction = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    cfg = QUALITY_CONFIG[prediction]

    st.markdown(f"""
    <div class="result-box" style="background:{cfg['bg']}; border: 2px solid {cfg['color']}60;">
        <div class="result-label" style="color:{cfg['color']}">
            {cfg['emoji']} Calidad {cfg['label']}
        </div>
        <div style="color:#9e8888; font-size:0.9rem; margin-top:0.4rem;">
            El modelo KNN clasifica este vino como de calidad <strong>{cfg['label'].lower()}</strong>
            — mueve los sliders para ver cómo cambia la predicción
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilidades por clase
    st.markdown('<div class="section-title">Probabilidades por clase</div>', unsafe_allow_html=True)

    col_l, col_m, col_h = st.columns(3)

    with col_l:
        st.metric("Baja 🍷", f"{proba[0]*100:.1f}%")
        st.progress(float(proba[0]))

    with col_m:
        st.metric("Media 🍷🍷", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

    with col_h:
        st.metric("Alta 🍷🍷🍷", f"{proba[2]*100:.1f}%")
        st.progress(float(proba[2]))

    # Comparación con promedios por clase (punto 2)
    st.markdown('<div class="section-title">Comparación con promedios por clase</div>', unsafe_allow_html=True)
    st.caption("Tus valores vs. el promedio histórico de cada categoría en el dataset")

    CLASS_MEANS = {
        "fixed acidity":        [7.23, 8.25, 8.85],
        "volatile acidity":     [0.72, 0.53, 0.40],
        "citric acid":          [0.17, 0.27, 0.38],
        "residual sugar":       [2.68, 2.54, 2.58],
        "chlorides":            [0.09, 0.09, 0.08],
        "free sulfur dioxide":  [11.0, 15.9, 13.6],
        "total sulfur dioxide": [40.1, 49.6, 33.0],
        "density":              [0.997, 0.997, 0.996],
        "pH":                   [3.40, 3.31, 3.37],
        "sulphates":            [0.57, 0.66, 0.74],
        "alcohol":              [9.55, 10.0, 11.5],
    }

    FEAT_LABELS = {
        "fixed acidity": "Fixed Acidity", "volatile acidity": "Volatile Acidity",
        "citric acid": "Citric Acid", "residual sugar": "Residual Sugar",
        "chlorides": "Chlorides", "free sulfur dioxide": "Free SO₂",
        "total sulfur dioxide": "Total SO₂", "density": "Density",
        "pH": "pH", "sulphates": "Sulphates", "alcohol": "Alcohol",
    }

    rows = []
    for feat, val in zip(FEATURE_NAMES, input_values):
        means = CLASS_MEANS[feat]
        rows.append({
            "Feature":     FEAT_LABELS[feat],
            "Tu valor":    round(val, 4),
            "⬛ Baja":     means[0],
            "🟧 Media":    means[1],
            "🟩 Alta":     means[2],
        })

    cmp_df = pd.DataFrame(rows).set_index("Feature")
    st.dataframe(cmp_df, use_container_width=True)

    st.markdown("""
    <div style="color:#9e8888; font-size:0.75rem; margin-top:0.3rem;">
    Valores calculados sobre el Red Wine Quality Dataset (UCI) · 1,599 muestras
    </div>
    """, unsafe_allow_html=True)

    # Tabla de valores raw
    with st.expander("📊 Ver valores actuales"):
        display_df = pd.DataFrame(
            {"Feature": FEATURE_NAMES, "Valor": input_values}
        ).set_index("Feature")
        st.dataframe(display_df, use_container_width=True)

except Exception as e:
    st.error(f"Error al predecir: {e}")


# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align:center; color:#9e8888; font-size:0.78rem;'>"
    "Steffano · 4Geeks Academy DS & ML Bootcamp · "
    "Modelo KNN entrenado con Red Wine Quality Dataset (UCI)"
    "</div>",
    unsafe_allow_html=True
)