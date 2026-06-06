"""
App Streamlit para interactuar con el modelo ONNX de predicción de cáncer de
mama. Permite cargar un caso de ejemplo o editar las características y obtener
una predicción. Cada predicción se registra en el archivo de texto del ambiente
(``predicciones_dev.txt`` / ``predicciones_prod.txt``) en S3.
"""

import json
import os

import streamlit as st

import s3_logger
from inference import predecir

ENV = os.environ.get("APP_ENV", "dev")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "baseline.json")

with open(BASELINE_PATH, encoding="utf-8") as f:
    META = json.load(f)

FEATURES = META["feature_names"]
CLASSES = META["classes"]  # ['malignant', 'benign']
CLASES_ES = {"malignant": "MALIGNO", "benign": "BENIGNO"}

# Casos reales del dataset para demostrar la app con un solo clic.
EJEMPLO_BENIGNO = [11.13, 16.62, 70.47, 381.1, 0.082, 0.038, 0.014, 0.014, 0.151,
                   0.061, 0.141, 0.967, 0.968, 9.704, 0.006, 0.006, 0.009, 0.006,
                   0.02, 0.002, 11.68, 20.29, 74.35, 421.1, 0.103, 0.062, 0.046,
                   0.04, 0.238, 0.071]
EJEMPLO_MALIGNO = [19.55, 28.77, 133.6, 1207.0, 0.093, 0.206, 0.178, 0.114, 0.189,
                   0.062, 0.843, 1.199, 7.158, 106.4, 0.006, 0.048, 0.039, 0.015,
                   0.019, 0.005, 25.05, 36.27, 178.6, 1926.0, 0.128, 0.533, 0.425,
                   0.194, 0.282, 0.101]


def _cargar_ejemplo(valores):
    for i, v in enumerate(valores):
        st.session_state[f"f_{i}"] = float(v)


st.set_page_config(page_title=f"Predictor [{ENV}]", page_icon="🔬", layout="wide")

st.title("🔬 Predictor de cáncer de mama")
c_env, c_info = st.columns([1, 4])
c_env.metric("Ambiente", ENV.upper())
c_info.caption(
    "Modelo en formato **ONNX** desplegado mediante CI/CD. El modelo se obtiene "
    "desde S3 durante el pipeline; cada predicción se registra en "
    f"`predicciones_{ENV}.txt` (S3)."
)

# Valores iniciales (caso benigno) la primera vez que se abre la app.
for i, v in enumerate(EJEMPLO_BENIGNO):
    st.session_state.setdefault(f"f_{i}", float(v))

st.write("**Cargar un caso de ejemplo:**")
b1, b2, b3 = st.columns(3)
b1.button("🟢 Caso benigno", on_click=_cargar_ejemplo, args=(EJEMPLO_BENIGNO,),
          use_container_width=True)
b2.button("🔴 Caso maligno", on_click=_cargar_ejemplo, args=(EJEMPLO_MALIGNO,),
          use_container_width=True)
b3.button("⚪ Reiniciar a ceros", on_click=_cargar_ejemplo,
          args=([0.0] * len(FEATURES),), use_container_width=True)

with st.expander("Ver / editar las 30 características de entrada", expanded=False):
    cols = st.columns(3)
    for i, feat in enumerate(FEATURES):
        cols[i % 3].number_input(feat, key=f"f_{i}", format="%.4f")

if st.button("🔮 Predecir", type="primary", use_container_width=True):
    valores = [st.session_state[f"f_{i}"] for i in range(len(FEATURES))]
    labels, probs = predecir(valores)
    label = int(labels[0])
    clase_es = CLASES_ES.get(CLASSES[label], CLASSES[label])
    prob = float(probs[0][label])

    if CLASSES[label] == "malignant":
        st.error(f"### Predicción: {clase_es}")
    else:
        st.success(f"### Predicción: {clase_es}")
    st.progress(prob, text=f"Confianza: {prob:.1%}")

    try:
        s3_logger.registrar(valores, clase_es, prob)
        st.caption(f"✅ Predicción registrada en `predicciones_{ENV}.txt` (S3).")
    except Exception as e:  # noqa: BLE001
        st.caption(f"⚠️ No se pudo registrar en S3: {e}")

st.divider()
st.caption(
    f"Clases del modelo: {', '.join(CLASES_ES.get(c, c) for c in CLASSES)} · "
    f"{len(FEATURES)} características de entrada · ambiente {ENV.upper()}."
)
