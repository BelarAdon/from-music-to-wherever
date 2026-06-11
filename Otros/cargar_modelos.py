import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from huggingface_hub import hf_hub_download
import joblib

repo_id = "TU_USUARIO/tu-repo-modelos"

scaler = joblib.load(hf_hub_download(repo_id, "modelos/scaler.pkl"))
umap_model = joblib.load(hf_hub_download(repo_id, "modelos/umap_model.pkl"))
kmeans = joblib.load(hf_hub_download(repo_id, "modelos/kmeans_umap.pkl"))
feature_cols = joblib.load(hf_hub_download(repo_id, "modelos/feature_cols.pkl"))

scaler = joblib.load(MODELS_DIR / "scaler.pkl")
umap_model = joblib.load(MODELS_DIR / "umap_model.pkl")
kmeans = joblib.load(MODELS_DIR / "kmeans_umap.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    
