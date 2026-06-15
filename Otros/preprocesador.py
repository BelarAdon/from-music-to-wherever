import numpy as np


def preprocesar_features(features_dict: dict, feature_cols: list, scaler) -> np.ndarray:

    x = np.array(
        [[features_dict[col] for col in feature_cols]],
        dtype=np.float32
    )
    return scaler.transform(x)
