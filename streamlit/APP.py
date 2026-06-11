import streamlit as st
import pandas as pd
import numpy as np
from Otros.cargar_modelos import scaler, kmeans, umap_model, feature_cols
from Otros.prediccion import predecir_mood_por_titulo, generar_outfit_recomendado, predecir_mood
from Otros.recomendador import distancia_umap, recomendar_por_cancion
from Otros.mood_imagenes import mood_imagenes
from Otros.descripciones_features import descripciones
import joblib
from huggingface_hub import hf_hub_download

# Cargar datos y modelos
df = pd.read_csv("Datos/data.csv")

# 2. CARGA DE MODELOS (AQUÍ ES DONDE VA TODO)
@st.cache_resource
def load_models():
    repo_id = "Eskarcho/modelo_streamlit"

    model1_path = hf_hub_download(repo_id, "modelos/features_cols.pkl")
    model2_path = hf_hub_download(repo_id, "modelos/umap_model.pkl")
    model3_path = hf_hub_download(repo_id, "modelos/kmeans_umap.pkl")
    model4_path = hf_hub_download(repo_id, "modelos/scaler.pkl")

    feature_cols = joblib.load(model1_path)
    umap_model = joblib.load(model2_path)
    kmeans = joblib.load(model3_path)
    scaler = joblib.load(model4_path)

    return feature_cols, umap_model, kmeans, scaler


feature_cols, umap_model, kmeans, scaler = load_models()
# Interfaz

st.title("🎧 ¿Qué escuchas? → Outfit Recommender")
st.write("Busca una canción o manipula sus audio-features para ver cómo cambia el mood 🎨👗")

tab_audio_features, tab_prediccion_cancion  = st.tabs([
    "🎛️ Predicción por audio features",
    "🎵 Predicción por canción"
    
])

# PREDICCIÓN INTERACTIVA POR AUDIO FEATURES

with tab_audio_features:

    st.header("🎛️ Ajusta las audio-features para ver cómo cambia el mood")

    titulo_input = st.text_input("Título de la canción", key="titulo_tab2")
    artista_input = st.text_input("Artista", key="artista_tab2")


    if "audio_features_base" not in st.session_state:
        st.session_state["audio_features_base"] = None
    if "audio_track_info" not in st.session_state:
        st.session_state["audio_track_info"] = None

    if st.button("Obtener audio features"):
        fila = df[
            (df["track_name"].str.lower() == titulo_input.lower().strip()) &
            (df["track_artist"].str.lower() == artista_input.lower().strip())
        ]

        if fila.empty:
            st.error("No se encontró la canción en el dataset.")
            st.session_state["audio_features_base"] = None
            st.session_state["audio_track_info"] = None
        else:
            fila = fila.iloc[0]
            st.session_state["audio_features_base"] = fila[feature_cols].to_dict()
            st.session_state["audio_track_info"] = {
                "title": fila["track_name"],
                "artist": fila["track_artist"]
            }
            st.success("✔ Audio-features cargadas. Ajusta los sliders para ver cómo cambia el mood.")

    if st.session_state["audio_features_base"] is not None:

        base_features = st.session_state["audio_features_base"]
        track_info = st.session_state["audio_track_info"]

        st.markdown(
            f"**Canción actual:** {track_info['title']} – {track_info['artist']}"
        )

        sliders = {}

        for col in feature_cols:
            base_val = float(base_features[col])
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            step = (max_val - min_val) / 100 if max_val != min_val else 0.01

            sliders[col] = st.slider(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=base_val,
                step=step,
                key=f"slider_{col}" 
            )

            st.caption(descripciones.get(col, "Sin descripción disponible."))

        #  FEATURES EDITADAS POR EL USUARIO

        features_dict = {col: sliders[col] for col in feature_cols}

     
        #  PREDICCIÓN DEL MOOD
        cluster_pred, mood = predecir_mood(features_dict, feature_cols, scaler, umap_model, kmeans)

        imagen_mood = mood_imagenes.get(mood, None)

        st.markdown(
            f"""
            <h1 style='
                text-align:center;
                font-size: 48px;
                font-weight: 800;
                color: #FF4B4B;
                margin-top: 20px;
            '>{mood}</h1>
            """,
            unsafe_allow_html=True
        )

        if imagen_mood:
            st.image(imagen_mood, use_container_width=True)


        resultado = generar_outfit_recomendado(cluster_pred)

        #  RECOMENDACIÓN BASADA EN FEATURES ACTUALES
        st.markdown("### 🎧 Canciones similares según tus audio-features")

        vec = np.array([features_dict[col] for col in feature_cols], dtype=float)
        scaled = scaler.transform([vec])
        emb = umap_model.transform(scaled)[0]

        df["distancia_temp"] = df.apply(
            lambda row: np.linalg.norm(
                emb - row[[f"umap_{i}" for i in range(10)]].values.astype(float)
            ),
            axis=1
        )

        recomendadas = df.sort_values("distancia_temp").head(10)

        st.dataframe(recomendadas[["track_name", "track_artist"]].rename(columns={'track_name': 'Título', 'track_artist': 'Artista'}))


# PREDICCIÓN POR CANCIÓN

key="titulo_tab1"
key="titulo_tab2"
key="artista_tab1"
key="artista_tab2"

with tab_prediccion_cancion:

    titulo = st.text_input("Título de la canción", key="titulo_tab1")
    artista = st.text_input("Artista (opcional)", key="artista_tab1")

    estacion = st.selectbox("Estación del año:", ["primavera", "verano", "otoño", "invierno"])
    clima = st.selectbox("Clima actual:", ["sol", "lluvia", "frio", "calor"])
    estilo = st.selectbox("Estilo personal:", ["femenino", "masculino", "unisex", "streetwear","minimal","edgy"])

    if st.button("Buscar y recomendar"):

        resultado = predecir_mood_por_titulo(
            df, titulo, artista, feature_cols, scaler, umap_model, kmeans,
            estacion=estacion, clima=clima, estilo=estilo
        )

        if "error" in resultado:
            st.error(resultado["error"])
        else:

            # Encabezado
            st.subheader(f"🎵 {resultado['title']} — {resultado['artist']}")
            st.write(f"**Mood detectado:** {resultado['mood']}  \n**Cluster:** {resultado['cluster']}")

            # -------------------
            # OUTFITS
            # -------------------
            outfit_final = resultado["outfit"]["outfit_final"]

            st.markdown("### 👗 Outfit recomendado final")
            st.write("**Prendas:**", outfit_final["prendas"])
            st.write("**Accesorios:**", outfit_final["accesorios"])
            st.write("📌", outfit_final["justificacion"])


            # PALETA
            st.markdown("### 🎨 Paleta de colores asociada al mood")
            palette = resultado["outfit"]["paleta_colores"]

            st.write(palette)
                        

            if resultado["outfit"].get("justificacion_paleta"):
                st.markdown("### 📌 Justificación de la paleta")
                st.write(resultado["outfit"]["justificacion_paleta"])






