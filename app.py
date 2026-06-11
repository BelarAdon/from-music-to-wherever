import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

from Otros.prediccion import predecir_mood_por_titulo, generar_outfit_recomendado
from Otros.mood_imagenes import mood_imagenes


repo_id = "Eskarcho/modelo_streamlit"


# =========================
# DATASET (OPTIMIZADO PARQUET)
# =========================
@st.cache_data
def load_data():
    path = hf_hub_download(repo_id, "Datos/dataset_final.parquet")
    return pd.read_parquet(path)

df = load_data()


# =========================
# MODELOS (SOLO UNA VEZ)
# =========================
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


# =========================
# UI
# =========================
st.title("🎧 Music → Outfit AI PRO")


tab1, tab2 = st.tabs(["🎛 Features", "🎵 Song"])


# =========================
# TAB 1 (OPTIMIZADO)
# =========================
with tab1:

    titulo = st.text_input("Título")
    artista = st.text_input("Artista")

    if st.button("Buscar"):

        row = df[
            (df["track_name"].str.lower() == titulo.lower().strip()) &
            (df["track_artist"].str.lower() == artista.lower().strip())
        ]

        if row.empty:
            st.error("No encontrada")
        else:
            row = row.iloc[0]

            st.success(f"{row['track_name']} — {row['track_artist']}")

            mood = row["mood"]
            st.markdown(f"## {mood}")
            st.image(mood_imagenes.get(mood))

            # =========================
            # RECOMENDACIÓN ULTRA OPTIMIZADA (NO COPY DF)
            # =========================

            emb = row[[f"embedding_{i}" for i in range(10)]].values.astype(np.float32)

            emb_matrix = np.vstack(df[[f"embedding_{i}" for i in range(10)]].values).astype(np.float32)

            # vectorizado (MUCHO más rápido y menos RAM)
            dists = np.linalg.norm(emb_matrix - emb, axis=1)

            idx = np.argsort(dists)[:10]

            st.dataframe(df.iloc[idx][["track_name", "track_artist"]])


# =========================
# TAB 2
# =========================
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
