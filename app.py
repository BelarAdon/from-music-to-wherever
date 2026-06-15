import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from sklearn.neighbors import NearestNeighbors
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
    df = df.reset_index(drop=True)

    # FIX 5: detectar columnas de embedding automáticamente en vez de hardcodear 10
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("embedding_")],
        key=lambda c: int(c.split("_")[1])
    )
    emb_matrix = df[emb_cols].to_numpy(dtype=np.float32)
    return df, emb_matrix

# FIX 3: índice de vecinos precalculado con sklearn — escala bien con datasets grandes
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
    scaler       = joblib.load(hf_hub_download(repo_id, "modelos/scaler.pkl"))
    kmeans       = joblib.load(hf_hub_download(repo_id, "modelos/kmeans_umap.pkl"))
    feature_cols = joblib.load(hf_hub_download(repo_id, "modelos/feature_cols.pkl"))
    return scaler, kmeans, feature_cols

# FIX 1: spinner explícito para que el usuario sepa que algo está cargando
with st.spinner("Cargando datos y modelos..."):
    df, emb_matrix   = load_data()
    nn_index         = build_index(emb_matrix)
    scaler, kmeans, feature_cols = load_models()

# =========================
# HELPERS
# =========================
def normalizar(s: str) -> str:
    """FIX 2: normaliza acentos, mayúsculas y espacios para búsqueda tolerante."""
    s = s.lower().strip()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

def buscar_fila(df: pd.DataFrame, titulo: str, artista: str):
    """Devuelve todas las filas que coinciden (puede haber duplicados)."""
    mask = (
        df["track_name"].apply(normalizar).str.contains(normalizar(titulo), regex=False) &
        df["track_artist"].apply(normalizar).str.contains(normalizar(artista), regex=False)
    )
    return df[mask]

# =========================
# UI
# =========================
st.title("🎧 Music → Outfit AI PRO")
tab1, tab2 = st.tabs(["🎛 Similar Songs", "👗 Outfit Recommender"])

# =========================
# TAB 1 - SIMILAR SONGS
# =========================
with tab1:
    titulo  = st.text_input("Título",   key="t1")
    artista = st.text_input("Artista",  key="t1a")

    if st.button("Buscar canción", key="btn_search"):
        if not titulo.strip():
            st.warning("Introduce al menos el título de la canción.")
        else:
            # FIX 2: búsqueda normalizada
            rows = buscar_fila(df, titulo, artista)

            if rows.empty:
                st.error("No encontrada. Prueba con otro título o artista.")
            else:
                # FIX 6: avisar si hay duplicados y dejar elegir
                if len(rows) > 1:
                    opciones = [
                        f"{r['track_name']} — {r['track_artist']} ({r['mood']})"
                        for _, r in rows.iterrows()
                    ]
                    sel = st.selectbox("Varias versiones encontradas, elige una:", opciones)
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

                # FIX 3: similitud con índice precalculado — O(log n) en vez de O(n)
                _, indices = nn_index.kneighbors([emb_matrix[idx]])
                nearest = indices[0][1:]  # excluir la propia canción
                st.dataframe(
                    df.iloc[nearest][["track_name", "track_artist", "mood"]]
                )

# =========================
# TAB 2 - OUTFIT
# =========================
with tab2:
    titulo2  = st.text_input("Título canción", key="t2")
    artista2 = st.text_input("Artista",        key="t2a")
    estacion = st.selectbox("Estación", ["primavera", "verano", "otoño", "invierno"])
    clima    = st.selectbox("Clima",    ["sol", "lluvia", "frio", "calor"])
    estilo   = st.selectbox(
        "Estilo",
        ["femenino", "masculino", "unisex", "streetwear", "minimal", "edgy"]
    )

    if st.button("Recomendar outfit", key="btn_outfit"):

        st.write("feature_cols del modelo:", feature_cols)
        st.write("columnas del df:", [c for c in df.columns if c in feature_cols])
        st.write("columnas que faltan:", [c for c in feature_cols if c not in df.columns])
        st.write("Número de feature_cols:", len(feature_cols))
        st.write("feature_cols completo:", feature_cols)
        
        if not titulo2.strip():
            st.warning("Introduce al menos el título de la canción.")
        else:
            # FIX 4: try/except para errores inesperados en el pipeline ML
            try:
                res = predecir_mood_por_titulo(
                    df,
                    titulo2,
                    artista2,
                    feature_cols,
                    scaler,
                    kmeans,
                    estacion=estacion,
                    clima=clima,
                    estilo=estilo,
                )
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.subheader(f"{res['title']} — {res['artist']}")
                    st.write(f"Mood: {res['mood']}")
                    outfit = res["outfit"]["outfit_final"]
                    st.markdown("### 👗 Outfit")
                    st.write("Prendas:",     outfit["prendas"])
                    st.write("Accesorios:",  outfit["accesorios"])
                    st.write(outfit["justificacion"])
            except Exception as e:
                st.error(f"Error inesperado al procesar la canción: {e}")
                st.stop()
