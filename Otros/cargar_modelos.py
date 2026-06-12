import joblib
from huggingface_hub import hf_hub_download

repo_id = "Eskarcho/modelo_streamlit"

# Variables globales (no cargadas aún)
scaler = None
kmeans = None
umap_model = None
feature_cols = None


def cargar_modelos():
    global scaler, kmeans, umap_model, feature_cols

    if scaler is None:
        scaler = joblib.load(
            hf_hub_download(repo_id, "modelos/scaler.pkl")
        )

    if kmeans is None:
        kmeans = joblib.load(
            hf_hub_download(repo_id, "modelos/kmeans_umap.pkl")
        )

    if umap_model is None:
        umap_model = joblib.load(
            hf_hub_download(repo_id, "modelos/umap_model.pkl")
        )

    if feature_cols is None:
        feature_cols = joblib.load(
            hf_hub_download(repo_id, "modelos/feature_cols.pkl")
        )

    return scaler, kmeans, umap_model, feature_cols
