import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

from Otros.prediccion import predecir_mood_por_titulo, generar_outfit_recomendado, predecir_mood
from Otros.recomendador import distancia_umap, recomendar_por_cancion
from Otros.mood_imagenes import mood_imagenes
from Otros.descripciones_features import descripciones


# =========================
# CONFIG
# =========================
repo_id = "Eskarcho/modelo_streamlit"


# =========================
# DATASET (CACHE DATA)
# =========================
@st.cache_data
def load_data():
    path = hf_hub_download(repo_id, "Datos/Data_Spotify_Features.csv")
    return pd.read_csv(path)


df = load_data()


# =========================
# MODELOS (CACHE RESOURCE)
# =========================
@st.cache_resource
def load_models():
    feature_cols = joblib.load(
        hf_hub_download(repo_id, "modelos/features_cols.pkl")
    )
    umap_model = joblib.load(
        hf_hub_download(repo_id, "modelos/umap_model.pkl")
    )
    kmeans = joblib.load(
        hf_hub_download(repo_id, "modelos/kmeans_umap.pkl")
    )
    scaler = joblib.load(
        hf_hub_download(repo_id, "modelos/scaler.pkl")
    )

    return feature_cols, umap_model, kmeans, scaler


feature_cols, umap_model, kmeans, scaler = load_models()


# =========================
# UI
# =========================
st.title("🎧 ¿Qué escuchas? → Outfit Recommender")
st.write("Busca una canción o manipula sus audio-features para ver cómo cambia el mood 🎨👗")


tab_audio_features, tab_prediccion_cancion = st.tabs([
    "🎛️ Predicción por audio features",
    "🎵 Predicción por canción"
])


# =========================
# TAB 1 - AUDIO FEATURES
# =========================
with tab_audio_features:

    st.header("🎛️ Ajusta las audio-features")

    titulo_input = st.text_input("Título de la canción", key="titulo_audio")
    artista_input = st.text_input("Artista", key="artista_audio")

    if "audio_features_base" not in st.session_state:
        st.session_state.audio_features_base = None
        st.session_state.audio_track_info = None

    if st.button("Obtener audio features"):

        fila = df[
            (df["track_name"].str.lower() == titulo_input.lower().strip()) &
            (df["track_artist"].str.lower() == artista_input.lower().strip())
        ]

        if fila.empty:
            st.error("No se encontró la canción.")
            st.session_state.audio_features_base = None
            st.session_state.audio_track_info = None
        else:
            fila = fila.iloc[0]

            st.session_state.audio_features_base = fila[feature_cols].to_dict()
            st.session_state.audio_track_info = {
                "title": fila["track_name"],
                "artist": fila["track_artist"]
            }

            st.success("Audio features cargadas ✔")


    if st.session_state.audio_features_base:

        base = st.session_state.audio_features_base
        info = st.session_state.audio_track_info

        st.markdown(f"**{info['title']} — {info['artist']}**")

        sliders = {}

        for col in feature_cols:
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            base_v = float(base[col])

            sliders[col] = st.slider(
                col,
                min_v,
                max_v,
                base_v,
                step=(max_v - min_v) / 100 if max_v != min_v else 0.01,
                key=f"slider_{col}"
            )

            st.caption(descripciones.get(col, ""))

        features_dict = sliders


        # =========================
        # PREDICCIÓN MOOD
        # =========================
        cluster, mood = predecir_mood(
            features_dict, feature_cols, scaler, umap_model, kmeans
        )

        st.markdown(
            f"<h1 style='text-align:center;color:#FF4B4B'>{mood}</h1>",
            unsafe_allow_html=True
        )

        if mood in mood_imagenes:
            st.image(mood_imagenes[mood])


        # =========================
        # RECOMENDACIÓN
        # =========================
        st.subheader("🎧 Canciones similares")

        vec = np.array([features_dict[c] for c in feature_cols], dtype=float)
        scaled = scaler.transform([vec])
        emb = umap_model.transform(scaled)[0]

        # VECTORIAL (MUCHO MÁS RÁPIDO QUE apply)
        umap_cols = [f"umap_{i}" for i in range(10)]
        emb_matrix = np.vstack(df[umap_cols].values)

        dists = np.linalg.norm(emb_matrix - emb, axis=1)

        df_rec = df.copy()
        df_rec["dist"] = dists

        recs = df_rec.sort_values("dist").head(10)

        st.dataframe(
            recs[["track_name", "track_artist"]]
            .rename(columns={"track_name": "Título", "track_artist": "Artista"})
        )


# =========================
# TAB 2 - CANCIÓN
# =========================
with tab_prediccion_cancion:

    titulo = st.text_input("Título", key="titulo_song")
    artista = st.text_input("Artista", key="artista_song")

    estacion = st.selectbox("Estación", ["primavera", "verano", "otoño", "invierno"])
    clima = st.selectbox("Clima", ["sol", "lluvia", "frio", "calor"])
    estilo = st.selectbox("Estilo", ["femenino", "masculino", "unisex", "streetwear", "minimal", "edgy"])

    if st.button("Buscar y recomendar"):

        resultado = predecir_mood_por_titulo(
            df, titulo, artista, feature_cols, scaler, umap_model, kmeans,
            estacion=estacion,
            clima=clima,
            estilo=estilo
        )

        if "error" in resultado:
            st.error(resultado["error"])
        else:

            st.subheader(f"{resultado['title']} — {resultado['artist']}")
            st.write(f"Mood: {resultado['mood']} | Cluster: {resultado['cluster']}")

            outfit = resultado["outfit"]["outfit_final"]

            st.markdown("### 👗 Outfit")
            st.write(outfit["prendas"])
            st.write(outfit["accesorios"])
            st.write(outfit["justificacion"])

            st.markdown("### 🎨 Paleta")
            st.write(resultado["outfit"]["paleta_colores"])
