import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from sklearn.neighbors import NearestNeighbors
from Otros.prediccion import predecir_mood_por_titulo
from Otros.mood_imagenes import mood_imagenes
from Otros.outfit_card import render_outfit_card

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
    df = df.reset_index(drop=True)
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("embedding_")],
        key=lambda c: int(c.split("_")[1])
    )
    emb_matrix = df[emb_cols].to_numpy(dtype=np.float32)
    return df, emb_matrix

@st.cache_resource(show_spinner=False)
def build_index(_emb_matrix):
    nn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
    nn.fit(_emb_matrix)
    return nn

# =========================
# MODELOS
# =========================
@st.cache_resource(show_spinner=False)
def load_models():
    kmeans = joblib.load(hf_hub_download(repo_id, "modelos/kmeans_umap.pkl"))
    return kmeans

# Columnas UMAP precalculadas en el dataset (evita cargar umap_model.pkl)
UMAP_COLS = [f"umap_{i}" for i in range(10)]

with st.spinner("Cargando datos y modelos..."):
    df, emb_matrix = load_data()
    nn_index       = build_index(emb_matrix)
    kmeans         = load_models()

# Validar que el dataset tiene las columnas umap necesarias
assert all(c in df.columns for c in UMAP_COLS), \
    f"Faltan columnas UMAP en el dataset: {[c for c in UMAP_COLS if c not in df.columns]}"

# =========================
# HELPERS
# =========================
def normalizar(s: str) -> str:
    s = s.lower().strip()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

def buscar_filas(df: pd.DataFrame, titulo: str, artista: str) -> pd.DataFrame:
    mask = df["track_name"].apply(normalizar).str.contains(normalizar(titulo), regex=False)
    if artista.strip():
        mask &= df["track_artist"].apply(normalizar).str.contains(normalizar(artista), regex=False)
    return df[mask]

# =========================
# SESSION STATE
# =========================
for key in ["tab1_activa", "tab1_titulo", "tab1_artista",
            "tab2_activa", "tab2_titulo", "tab2_artista",
            "tab2_estacion", "tab2_clima", "tab2_estilo"]:
    if key not in st.session_state:
        st.session_state[key] = False if key.endswith("activa") else ""

# =========================
# UI
# =========================
st.title("🎧 Music → Outfit AI PRO")
tab1, tab2 = st.tabs(["🎛 Similar Songs", "👗 Outfit Recommender"])

# =========================
# TAB 1 - SIMILAR SONGS
# =========================
with tab1:
    titulo  = st.text_input("Título",  key="t1")
    artista = st.text_input("Artista", key="t1a")

    if st.button("Buscar canción", key="btn_search"):
        if not titulo.strip():
            st.warning("Introduce al menos el título de la canción.")
        else:
            st.session_state.tab1_activa  = True
            st.session_state.tab1_titulo  = titulo
            st.session_state.tab1_artista = artista

    if st.session_state.tab1_activa:
        rows = buscar_filas(df, st.session_state.tab1_titulo, st.session_state.tab1_artista)

        if rows.empty:
            st.error("No encontrada. Prueba con otro título o artista.")
            st.session_state.tab1_activa = False
        else:
            if len(rows) > 1:
                opciones = [
                    f"{r['track_name']} — {r['track_artist']} ({r['mood']})"
                    for _, r in rows.iterrows()
                ]
                sel = st.selectbox("Varias versiones encontradas, elige una:", opciones, key="sel_tab1")
                idx = rows.index[opciones.index(sel)]
            else:
                idx = rows.index[0]

            song = df.loc[idx]
            st.success(f"{song['track_name']} — {song['track_artist']}")
            mood = song["mood"]
            st.markdown(f"## Mood: {mood}")

            img = mood_imagenes.get(mood)
            if img:
                st.image(img)
            else:
                st.warning(f"No hay imagen para mood: {mood}")

            _, indices = nn_index.kneighbors([emb_matrix[idx]])
            nearest = indices[0][1:]
            st.dataframe(df.iloc[nearest][["track_name", "track_artist", "mood"]])

# =========================
# TAB 2 - OUTFIT
# =========================
with tab2:
    titulo2  = st.text_input("Título canción", key="t2")
    artista2 = st.text_input("Artista",        key="t2a")
    estacion = st.selectbox("Estación", ["primavera", "verano", "otoño", "invierno"])
    clima    = st.selectbox("Clima",    ["sol", "lluvia", "frio", "calor"])
    estilo   = st.selectbox("Estilo",   ["femenino", "masculino", "unisex", "streetwear", "minimal", "edgy"])

    if st.button("Recomendar outfit", key="btn_outfit"):
        if not titulo2.strip():
            st.warning("Introduce al menos el título de la canción.")
        else:
            st.session_state.tab2_activa   = True
            st.session_state.tab2_titulo   = titulo2
            st.session_state.tab2_artista  = artista2
            st.session_state.tab2_estacion = estacion
            st.session_state.tab2_clima    = clima
            st.session_state.tab2_estilo   = estilo

    if st.session_state.tab2_activa:
        try:
            res = predecir_mood_por_titulo(
                df,
                st.session_state.tab2_titulo,
                st.session_state.tab2_artista,
                UMAP_COLS,
                kmeans,
                estacion=st.session_state.tab2_estacion,
                clima=st.session_state.tab2_clima,
                estilo=st.session_state.tab2_estilo,
            )
            if "error" in res:
                st.error(res["error"])
                st.session_state.tab2_activa = False
            else:
                render_outfit_card(res)
        except Exception as e:
            st.error(f"Error inesperado al procesar la canción: {e}")
            st.session_state.tab2_activa = False
