import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import gzip

# Cargar el modelo y las características de entrenamiento
model_filename = "decision_tree_classifier_anime.joblib.gz"

# Descomprimir y cargar el modelo
with gzip.open(model_filename, 'rb') as f:
    model = load(f)

# Cargar las características de entrenamiento
train_features = pd.read_csv('train_features.csv').values.flatten()

# Título de la aplicación
st.title('Análisis del Dataset de Anime')

# Subtítulo
st.header('Predicción del Nombre del Anime')

# Cargar los datos
@st.cache_data
def load_data():
    data = pd.read_csv('anime.csv')
    return data

data = load_data()

# Verificar los nombres de las columnas
st.write("Columnas del DataFrame:")
st.write(data.columns)

# Mostrar el DataFrame
st.write("Aquí están los datos del dataset de anime:")
st.write(data)

# Asegurarse de que la columna 'rating' existe (ajusta según tu interés)
if 'rating' not in data.columns:
    st.error("La columna 'rating' no se encuentra en el DataFrame.")
else:
    # Crear un gráfico de líneas con los datos
    st.header('Gráfico de líneas de los ratings')
    st.line_chart(data['rating'].dropna())  # Asegúrate de eliminar NaN si es necesario

    # Slider para seleccionar un valor de rating
    valor_seleccionado = st.slider('Selecciona un valor de rating', 0.0, 10.0, float(data['rating'].mean()))
    st.write('Valor seleccionado:', valor_seleccionado)

    # Selección de géneros para la predicción
    unique_genres = sorted(set(g for sublist in data['genre'].dropna().str.split(', ') for g in sublist))
    selected_genres = st.multiselect('Selecciona los géneros', unique_genres)

    if st.button('Predecir Nombre del Anime'):
        input_data = pd.DataFrame(columns=train_features)
        input_data.loc[0] = 0  # Inicializar con ceros

        # Asignar el valor seleccionado de rating
        input_data['rating'] = valor_seleccionado

        # Asignar los valores seleccionados a las características
        for genre in selected_genres:
            if genre in train_features:
                input_data[genre] = 1
            else:
                input_data[genre] = 0  # Asegúrate de agregar 0 si no se encuentra el género

        # Predicción
        predicted_anime = model.predict(input_data.fillna(0))[0]
        st.write('El nombre del anime predicho es:', predicted_anime)
