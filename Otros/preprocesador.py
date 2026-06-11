import numpy as np
from Otros.cargar_modelos import scaler, umap_model, kmeans, feature_cols

def preprocesar_features(features_dict):

    dtype = kmeans.cluster_centers_.dtype  
    x = np.array([[features_dict[col] for col in feature_cols]], dtype=dtype)
    x_scaled = scaler.transform(x.astype(dtype, copy=False))
    x_umap = umap_model.transform(x_scaled)
    x_umap = x_umap.astype(dtype, copy=False)

    return x_umap

