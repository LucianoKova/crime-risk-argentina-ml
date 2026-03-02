import streamlit as st
import pandas as pd
import joblib

st.title("Predicción de Uso de Arma en Delitos - CABA")

# Cargar modelo
model = joblib.load("models/random_forest_model.pkl")

# Inputs del usuario
franja = st.slider("Franja horaria (0-23)", 0, 23, 18)
comuna = st.selectbox("Comuna", list(range(1, 16)))
uso_moto = st.selectbox("¿Uso de moto?", ["NO", "SI"])
tipo = st.selectbox("Tipo de delito", ["Robo", "Hurto", "Lesiones", "Vialidad"])

# Crear DataFrame
input_data = pd.DataFrame({
    "franja": [franja],
    "comuna": [comuna],
    "uso_moto": [uso_moto],
    "tipo": [tipo]
})

# One-hot encoding igual que entrenamiento
input_data = pd.get_dummies(input_data)

# Alinear columnas con modelo
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predecir"):
    proba = model.predict_proba(input_data)[0][1]
    
    st.write(f"Probabilidad estimada de uso de arma: {proba:.2%}")
    
    if proba > 0.5:
        st.error("Alta probabilidad de uso de arma")
    else:
        st.success("Baja probabilidad de uso de arma")