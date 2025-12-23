import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "modelos" 

scaler = joblib.load(MODELS_DIR / "scaler.pkl")
umap_model = joblib.load(MODELS_DIR / "umap_model.pkl")
kmeans = joblib.load(MODELS_DIR / "kmeans_umap.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    
