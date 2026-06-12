import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

from Otros.prediccion import predecir_mood_por_titulo, generar_outfit_recomendado
from Otros.mood_imagenes import mood_imagenes

repo_id = "Eskarcho/modelo_streamlit"


# ==========================================
# DATASET Y MATRIZ (OPTIMIZADO EN CACHÉ)
# ==========================================
@st.cache_data
def load_data():
    path = hf_hub_download(repo_id, "Datos/dataset_final.parquet")
    data = pd.read_parquet(path)
    
    # Extraemos las 10 columnas de embeddings directas a Numpy (Evita np.vstack más adelante)
    columnas_emb = [f"embedding_{i}" for i in range(10)]
    matrix = data[columnas_emb].to_numpy().astype(np.float32)
    
    return data, matrix

# Cargamos el DataFrame y la matriz global libre de duplicados en RAM
df, emb_matrix = load_data()


# ==========================================
# MODELOS (SOLO UNA VEZ)
# ==========================================
@st.cache_resource
def load_models():
    scaler_path = hf_hub_download(repo_id, "modelos/scaler.pkl")
    kmeans_path = hf_hub_download(repo_id, "modelos/kmeans_umap.pkl")
    umap_path = hf_hub_download(repo_id, "modelos/umap_model.pkl")
    features_path = hf_hub_download(repo_id, "modelos/feature_cols.pkl")

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    umap_model = joblib.load(umap_path)
    feature_cols = joblib.load(features_path)

    return scaler, kmeans, umap_model, feature_cols


scaler, kmeans, umap_model, feature_cols = load_models()


# ==========================================
# UI
# ==========================================
st.title("🎧 Music → Outfit AI PRO")

tab1, tab2 = st.tabs(["🎛 Features", "🎵 Song"])


# ==========================================
# TAB 1 (OPTIMIZADO CONTRA MEMORY ERROR)
# ==========================================
with tab1:

    titulo = st.text_input("Título")
    artista = st.text_input("Artista")

    if st.button("Buscar"):

        row_mask = (df["track_name"].str.lower() == titulo.lower().strip()) & \
                   (df["track_artist"].str.lower() == artista.lower().strip())
        
        row = df[row_mask]

        if row.empty:
            st.error("No encontrada")
        else:
            # Conseguimos la posición (índice) exacto de la fila encontrada
            idx_encontrado = row.index[0]
            row_data = row.iloc[0]

            st.success(f"{row_data['track_name']} — {row_data['track_artist']}")

            mood = row_data["mood"]
            st.markdown(f"## {mood}")
            st.image(mood_imagenes.get(mood))

            # ==========================================
            # RECOMENDACIÓN CON CERO DUPLICACIÓN DE MEMORIA
            # ==========================================
            
            # Extraemos el vector de la canción usando el índice directo de la matriz precargada
            emb = emb_matrix[idx_encontrado]

            # Operación matemática directa (Broadcasting rápido sin devorar RAM)
            dists = np.linalg.norm(emb_matrix - emb, axis=1)

            # Ordenamos y extraemos los 10 índices más cercanos
            idx_mas_cercanos = np.argsort(dists)[:10]

            st.dataframe(df.iloc[idx_mas_cercanos][["track_name", "track_artist"]])


# ==========================================
# TAB 2
# ==========================================
with tab2:

    titulo = st.text_input("Título canción", key="t2")
    artista = st.text_input("Artista", key="t2a")

    estacion = st.selectbox("Estación", ["primavera", "verano", "otoño", "invierno"])
    clima = st.selectbox("Clima", ["sol", "lluvia", "frio", "calor"])
    estilo = st.selectbox("Estilo", ["femenino", "masculino", "unisex", "streetwear", "minimal", "edgy"])

    if st.button("Recomendar outfit"):

        res = predecir_mood_por_titulo(
            df,
            titulo,
            artista,
            feature_cols,
            scaler,
            umap_model,
            kmeans,
            estacion=estacion,
            clima=clima,
            estilo=estilo
        )

        if "error" in res:
            st.error(res["error"])
        else:

            st.subheader(f"{res['title']} — {res['artist']}")
            st.write(f"Mood: {res['mood']}")

            outfit = res["outfit"]["outfit_final"]

            st.markdown("### 👗 Outfit")
            st.write(outfit["prendas"])
            st.write(outfit["accesorios"])
            st.write(outfit["justificacion"])
