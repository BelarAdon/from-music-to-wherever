import unicodedata
import pandas as pd
from Otros.preprocesador import preprocesar_features
from Otros.outfit_mapping import outfit_mapping
from Otros.palette_mapping import palette_mapping


# =========================
# HELPERS
# =========================
def normalizar(s: str) -> str:
    """Normaliza acentos, mayúsculas y espacios para búsqueda tolerante."""
    s = s.lower().strip()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def buscar_cancion(df: pd.DataFrame, titulo: str, artista: str = None):
    """
    Devuelve la primera fila coincidente o None.
    Usa búsqueda parcial normalizada (tolerante a acentos y mayúsculas).
    """
    mask = df["track_name"].apply(normalizar).str.contains(
        normalizar(titulo), regex=False
    )
    if artista:
        mask &= df["track_artist"].apply(normalizar).str.contains(
            normalizar(artista), regex=False
        )
    df_filtrado = df[mask]
    if df_filtrado.empty:
        return None
    return df_filtrado.iloc[0]


# =========================
# PREDICCIÓN
# =========================
def predecir_mood(fila, umap_cols, kmeans):
    emb = preprocesar_features(fila, umap_cols)
    cluster = int(kmeans.predict(emb)[0])
    mood = outfit_mapping[cluster]["mood_name"]
    return cluster, mood


def combinar_outfits(
    base: dict,
    estilo_conf: dict = None,
    estacion_conf: dict = None,
    clima_conf: dict = None,
) -> dict:
    """Combina prendas y accesorios de las distintas capas de configuración."""
    prendas        = set(base.get("prendas", []))
    accesorios     = set(base.get("accesorios", []))
    justificaciones = [base.get("justificacion", "")]

    for conf in [estilo_conf, estacion_conf, clima_conf]:
        if conf:
            prendas.update(conf.get("prendas", []))
            accesorios.update(conf.get("accesorios", []))
            j = conf.get("justificacion", "")
            if j:
                justificaciones.append(j)

    return {
        "prendas":       list(prendas),
        "accesorios":    list(accesorios),
        "justificacion": " ".join(justificaciones),
    }


def generar_outfit_recomendado(
    cluster: int,
    estacion: str = None,
    clima: str = None,
    estilo: str = None,
) -> dict:
    info  = outfit_mapping[cluster]
    mood  = info["mood_name"]
    paleta_info = palette_mapping.get(mood, {})

    base         = info["outfit_base"]
    estilo_conf  = info["por_estilo"].get(estilo)   if estilo   else None
    estacion_conf= info["por_estacion"].get(estacion.lower()) if estacion else None
    clima_conf   = info["por_clima"].get(clima.lower())       if clima    else None

    outfit_final = combinar_outfits(
        base,
        estilo_conf=estilo_conf,
        estacion_conf=estacion_conf,
        clima_conf=clima_conf,
    )

    return {
        "mood":                mood,
        "paleta_colores":      paleta_info.get("colores", []),
        "justificacion_paleta":paleta_info.get("justificacion", ""),
        "outfit_final":        outfit_final,
    }


def predecir_mood_por_titulo(df, titulo, artista, umap_cols, kmeans, estacion=None, clima=None, estilo=None):
    fila = buscar_cancion(df, titulo, artista)
    if fila is None:
        return {"error": "Canción no encontrada"}
    cluster, mood = predecir_mood(fila, umap_cols, kmeans)
    outfit = generar_outfit_recomendado(cluster, estacion=estacion, clima=clima, estilo=estilo)
    return {
        "title":   fila["track_name"],
        "artist":  fila.get("track_artist", "Desconocido"),
        "mood":    mood,
        "cluster": cluster,
        "outfit":  outfit,
    }

    fila = buscar_cancion(df, titulo, artista)
    if fila is None:
        return {"error": "Canción no encontrada"}

    features_dict = fila[feature_cols].to_dict()

    # FIX: firma correcta sin umap_model
    cluster, mood = predecir_mood(features_dict, feature_cols, scaler, kmeans)

    outfit = generar_outfit_recomendado(
        cluster,
        estacion=estacion,
        clima=clima,
        estilo=estilo,
    )

    return {
        "title":   fila["track_name"],
        "artist":  fila.get("track_artist", "Desconocido"),
        "mood":    mood,
        "cluster": cluster,
        "outfit":  outfit,
    }
