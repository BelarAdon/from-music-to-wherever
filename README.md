# 🎧 From Music to Outfit

Proyecto final de Machine Learning basado en el análisis de audio-features musicales para detectar moods emocionales y recomendar canciones por similitud acústica.

## 🧠 Descripción
Este proyecto utiliza técnicas de Machine Learning no supervisado para:
- Representar canciones en un espacio acústico latente
- Agruparlas según similitud sonora
- Recomendar canciones similares
- Explorar la traducción del mood musical a una capa estética (proof of concept)

El enfoque se basa exclusivamente en audio-features, sin utilizar letras, géneros ni popularidad.

## ⚙️ Tecnologías utilizadas
- Python
- Pandas, NumPy
- Scikit-learn
- UMAP
- Streamlit

## 🧩 Pipeline
Audio-features → RobustScaler → UMAP → KMeans → Mood → Recomendación

## 🎛️ Aplicación
La aplicación desarrollada con Streamlit permite:
- Buscar una canción y predecir su mood
- Ajustar audio-features para observar cambios en la predicción
- Obtener recomendaciones musicales

## 📦 Modelos entrenados

El archivo `umap_model.pkl` no se incluye en el repositorio debido a su tamaño.

Para reproducir el proyecto:
1. Ejecutar el notebook de entrenamiento
2. Esto generará automáticamente los modelos necesarios en la carpeta `/modelos`

El pipeline es completamente reproducible.


## 🚧 Estado del proyecto
- ✅ Core de Machine Learning validado
- ⚠️ Capa estética en desarrollo (proof of concept)

## 📄 Memoria
La memoria completa del proyecto se encuentra en la carpeta `/memoria`.

## 👩‍💻 Autora
Marina Xiuping Garrido Castaño
Proyecto Final Bootcamp Machine Learning
