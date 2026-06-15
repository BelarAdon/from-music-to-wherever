import numpy as np

import numpy as np

def preprocesar_features(fila, umap_cols):
    return np.array([[fila[col] for col in umap_cols]], dtype=np.float32)
