import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===============================
# CONFIGURACIÓN
# ===============================
st.set_page_config(
    page_title="Predicción de Uso de Arma - CABA",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 Predicción de Uso de Arma en Delitos - CABA")

st.markdown(
    "Modelo Random Forest entrenado con datos oficiales de delitos CABA 2022. "
    "El sistema estima la probabilidad de que un hecho delictivo involucre uso de arma."
)

# ===============================
# CARGA DEL MODELO
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_small.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Modelo no encontrado en el repositorio.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ===============================
# INPUTS DEL USUARIO
# ===============================
st.subheader("📊 Datos del delito")

franja = st.slider("Franja horaria (0–23)", 0, 23, 18)
comuna = st.selectbox("Comuna", list(range(1, 16)))
uso_moto = st.selectbox("¿Uso de moto?", ["NO", "SI"])
tipo = st.selectbox("Tipo de delito", ["Robo", "Hurto", "Lesiones", "Vialidad"])

input_data = pd.DataFrame({
    "franja": [franja],
    "comuna": [comuna],
    "uso_moto": [uso_moto],
    "tipo": [tipo]
})

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# ===============================
# PREDICCIÓN
# ===============================
if st.button("Predecir riesgo"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("🔴 Alta probabilidad de uso de arma")
    else:
        st.success("🟢 Baja probabilidad de uso de arma")

    st.metric("Probabilidad estimada", f"{probability*100:.2f}%")
    st.progress(int(probability * 100))

    if probability < 0.30:
        st.info("Nivel de riesgo: Bajo")
    elif probability < 0.60:
        st.warning("Nivel de riesgo: Medio")
    else:
        st.error("Nivel de riesgo: Alto")

# ===============================
# MÉTRICAS DEL MODELO
# ===============================
st.markdown("---")
st.subheader("📊 Métricas del Modelo")

st.markdown(
"""
- Accuracy aproximada: ~0.89  
- Recall clase 'SI' (uso de arma): ~0.29  
- Dataset con fuerte desbalance de clases  
- Se utilizó `class_weight="balanced"` para compensar minoría  
"""
)

# ===============================
# IMPORTANCIA DE VARIABLES
# ===============================
st.markdown("---")
st.subheader("📈 Variables más importantes")

importances = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_
).sort_values(ascending=True).tail(8)

# Limpiar nombres técnicos
def clean_name(name):
    name = name.replace("tipo_", "")
    name = name.replace("uso_moto_", "Moto ")
    name = name.replace("franja", "Franja horaria")
    name = name.replace("comuna", "Comuna")
    return name

importances.index = [clean_name(col) for col in importances.index]

# Gráfico horizontal más profesional
st.bar_chart(importances)

# ===============================
# EXPLICACIÓN TÉCNICA
# ===============================
st.markdown("---")
st.subheader("🧠 Consideraciones Técnicas")

st.markdown(
"""
El problema presenta un fuerte desbalance de clases 
(la mayoría de los delitos no involucran uso de arma).

Para mitigar esto:

- Se utilizó `class_weight="balanced"`
- Se priorizó recall sobre accuracy pura
- Se redujo el dataset para despliegue eficiente en cloud
"""
)