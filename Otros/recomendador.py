import numpy as np

def distancia_umap(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def recomendar_por_cancion(df, titulo, artista, top_n=10):

    fila = df[
        (df["track_name"].str.lower() == titulo.lower()) &
        (df["track_artist"].str.lower() == artista.lower())
    ]
    
    if fila.empty:
        return {"error": "Canción no encontrada"}
    
    fila = fila.iloc[0]
    
    base_vec = fila[[f"umap_{i}" for i in range(10)]].values.astype(float)

    df["distancia"] = df.apply(
        lambda row: distancia_umap(
            base_vec,
            row[[f"umap_{i}" for i in range(10)]].values.astype(float)
        ), 
        axis=1
    )

    df_filtrado = df.drop(fila.name)

    recomendadas = df_filtrado.sort_values("distancia").head(top_n)

    return recomendadas[[
        "track_name",
        "track_artist",
        "tipo_clusters",
        "distancia"
    ]]
