import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

from Otros.prediccion import predecir_mood_por_titulo
from Otros.mood_imagenes import mood_imagenes


# =========================
# REPO
# =========================
repo_id = "Eskarcho/modelo_streamlit"


# =========================
# DATASET + EMBEDDINGS
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    path = hf_hub_download(repo_id, "Datos/dataset_final.parquet")
    df = pd.read_parquet(path)

    # IMPORTANTE: evitar desalineación embeddings-index
    df = df.reset_index(drop=True)

    emb_cols = [f"embedding_{i}" for i in range(10)]
    emb_matrix = df[emb_cols].to_numpy(dtype=np.float32)

    return df, emb_matrix


df, emb_matrix = load_data()


# =========================
# MODELOS
# =========================
@st.cache_resource(show_spinner=False)
def load_models():
    scaler = joblib.load(hf_hub_download(repo_id, "modelos/scaler.pkl"))
    kmeans = joblib.load(hf_hub_download(repo_id, "modelos/kmeans_umap.pkl"))
    feature_cols = joblib.load(hf_hub_download(repo_id, "modelos/feature_cols.pkl"))

    return scaler, kmeans, feature_cols


scaler, kmeans, feature_cols = load_models()


# =========================
# UI
# =========================
st.title("🎧 Music → Outfit AI PRO")

tab1, tab2 = st.tabs(["🎛 Similar Songs", "👗 Outfit Recommender"])


# =========================
# TAB 1 - SIMILAR SONGS
# =========================
with tab1:

    titulo = st.text_input("Título", key="t1")
    artista = st.text_input("Artista", key="t1a")

    if st.button("Buscar canción", key="btn_search"):

        mask = (
            (df["track_name"].str.lower().str.strip() == titulo.lower().strip()) &
            (df["track_artist"].str.lower().str.strip() == artista.lower().strip())
        )

        row = df[mask]

        if row.empty:
            st.error("No encontrada")
        else:
            idx = row.index[0]
            song = row.iloc[0]

            st.success(f"{song['track_name']} — {song['track_artist']}")

            mood = song["mood"]
            st.markdown(f"## Mood: {mood}")

            img = mood_imagenes.get(mood)
            if img:
                st.image(img)
            else:
                st.warning(f"No hay imagen para mood: {mood}")

            # =========================
            # SIMILITUD (OPTIMIZADA)
            # =========================
            emb = emb_matrix[idx]

            dists = np.linalg.norm(emb_matrix - emb, axis=1)
            nearest = np.argsort(dists)[1:11]

            st.dataframe(
                df.iloc[nearest][["track_name", "track_artist", "mood"]]
            )


# =========================
# TAB 2 - OUTFIT
# =========================
with tab2:

    titulo2 = st.text_input("Título canción", key="t2")
    artista2 = st.text_input("Artista", key="t2a")

    estacion = st.selectbox("Estación", ["primavera", "verano", "otoño", "invierno"])
    clima = st.selectbox("Clima", ["sol", "lluvia", "frio", "calor"])
    estilo = st.selectbox(
        "Estilo",
        ["femenino", "masculino", "unisex", "streetwear", "minimal", "edgy"]
    )

    if st.button("Recomendar outfit", key="btn_outfit"):

        res = predecir_mood_por_titulo(
            df,
            titulo2,
            artista2,
            feature_cols,
            scaler,
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
            st.write("Prendas:", outfit["prendas"])
            st.write("Accesorios:", outfit["accesorios"])
            st.write(outfit["justificacion"])
