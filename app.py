import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

from Otros.prediccion import predecir_mood_por_titulo, generar_outfit_recomendado
from Otros.mood_imagenes import mood_imagenes
from Otros.descripciones_features import descripciones


# =========================
# CONFIG
# =========================
repo_id = "Eskarcho/modelo_streamlit"


# =========================
# DATASET (OPTIMIZADO)
# =========================
@st.cache_data
def load_data():
    path = hf_hub_download(repo_id, "Datos/dataset_final.parquet")
    return pd.read_parquet(path)

df = load_data()


# =========================
# EMBEDDINGS MATRIX (CRÍTICO PARA RENDIMIENTO)
# =========================
@st.cache_resource
def get_embedding_matrix(df):
    return np.vstack(df[[f"embedding_{i}" for i in range(10)]].values)

emb_matrix = get_embedding_matrix(df)


# =========================
# MODELOS LIGEROS
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load(hf_hub_download(repo_id, "modelos/scaler.pkl"))
    kmeans = joblib.load(hf_hub_download(repo_id, "modelos/kmeans.pkl"))
    return scaler, kmeans

scaler, kmeans = load_models()


# =========================
# UI
# =========================
st.title("🎧 Music → Outfit AI (PRO VERSION)")


tab1, tab2 = st.tabs(["🎛 Audio features", "🎵 Song search"])


# =========================
# TAB 1 - SEARCH + RECOMMENDATION
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

            # 🎯 MOOD YA PRECALCULADO
            mood = row["mood"]
            st.markdown(f"## {mood}")
            st.image(mood_imagenes.get(mood))

            # =========================
            # RECOMENDADOR ULTRA RÁPIDO
            # =========================

            emb = row[[f"embedding_{i}" for i in range(10)]].values

            # vectorización (sin copy de df)
            dists = np.linalg.norm(emb_matrix - emb, axis=1)

            recs = df.copy()
            recs["dist"] = dists

            recs = recs.sort_values("dist").head(10)

            st.dataframe(
                recs[["track_name", "track_artist"]]
                .rename(columns={"track_name": "Título", "track_artist": "Artista"})
            )


# =========================
# TAB 2 - OUTFIT
# =========================
with tab2:

    titulo = st.text_input("Título canción", key="t2")
    artista = st.text_input("Artista", key="t2a")

    estacion = st.selectbox("Estación", ["primavera", "verano", "otoño", "invierno"])
    clima = st.selectbox("Clima", ["sol", "lluvia", "frio", "calor"])
    estilo = st.selectbox(
        "Estilo",
        ["femenino", "masculino", "unisex", "streetwear", "minimal", "edgy"]
    )

    if st.button("Recomendar outfit"):

        res = predecir_mood_por_titulo(
            df,
            titulo,
            artista,
            None,
            None,
            None,
            None,
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
            st.write("Prendas:", outfit["prendas"])
            st.write("Accesorios:", outfit["accesorios"])
            st.write("Justificación:", outfit["justificacion"])
