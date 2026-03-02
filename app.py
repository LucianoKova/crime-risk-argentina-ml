import os
import joblib
import pandas as pd
import streamlit as st

# ===============================
# CONFIGURACIÓN DE PÁGINA
# ===============================
st.set_page_config(
    page_title="Predicción de Uso de Arma - CABA",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 Predicción de Uso de Arma en Delitos - CABA")
st.markdown("Modelo de Machine Learning (Random Forest) entrenado con datos 2022.")

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

# ===============================
# PREPARACIÓN DE DATOS
# ===============================
input_data = pd.DataFrame({
    "franja": [franja],
    "comuna": [comuna],
    "uso_moto": [uso_moto],
    "tipo": [tipo]
})

# One-hot encoding igual que en entrenamiento
input_data = pd.get_dummies(input_data)

# Alinear columnas con modelo
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# ===============================
# PREDICCIÓN
# ===============================
if st.button("Predecir riesgo"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"🔴 Alta probabilidad de uso de arma")
    else:
        st.success(f"🟢 Baja probabilidad de uso de arma")

    st.metric("Probabilidad estimada", f"{probability*100:.2f}%")
