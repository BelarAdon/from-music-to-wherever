import numpy as np
from Otros.cargar_modelos import scaler, kmeans, feature_cols
def preprocesar_features(features_dict):

    x = np.array([[features_dict[col] for col in feature_cols]], dtype=np.float32)

    x_scaled = scaler.transform(x)
    
    return x_scaled
