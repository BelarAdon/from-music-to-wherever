import unicodedata
import pandas as pd
import numpy as np
from Otros.preprocesador import preprocesar_features
from Otros.outfit_mapping import outfit_mapping
from Otros.palette_mapping import palette_mapping


def normalizar(s: str) -> str:
    s = s.lower().strip()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def buscar_cancion(df: pd.DataFrame, titulo: str, artista: str = None):
    mask = df["track_name"].apply(normalizar).str.contains(normalizar(titulo), regex=False)
    if artista and artista.strip():
        mask &= df["track_artist"].apply(normalizar).str.contains(normalizar(artista), regex=False)
    df_filtrado = df[mask]
    if df_filtrado.empty:
        return None
    return df_filtrado.iloc[0]


def predecir_mood(fila: pd.Series, umap_cols: list, kmeans) -> tuple:
    emb = preprocesar_features(fila, umap_cols)
    cluster = int(kmeans.predict(emb)[0])
    mood = outfit_mapping[cluster]["mood_name"]
    return cluster, mood


def combinar_outfits(base, estilo_conf=None, estacion_conf=None, clima_conf=None):
    """
    Combina outfit con prioridad de capas y límites:
    - Estilo define la identidad: reemplaza prendas del base, no acumula
    - Estación añade contexto: sustituye 1 prenda del base si hay solapamiento funcional
    - Clima aporta solo 1 elemento funcional extra (impermeable, abrigo, etc.)
    - Límite: máximo 4 prendas y 3 accesorios en el resultado final
    - Justificación: solo la más relevante (estilo > estación > base)
    """
    MAX_PRENDAS    = 4
    MAX_ACCESORIOS = 3

    # Base como punto de partida (lista ordenada, no set)
    prendas    = list(base.get("prendas", []))
    accesorios = list(base.get("accesorios", []))
    justificacion = base.get("justificacion", "")

    # Capa 1: ESTILO — reemplaza prendas del base, tiene prioridad total
    if estilo_conf:
        estilo_prendas    = estilo_conf.get("prendas", [])
        estilo_accesorios = estilo_conf.get("accesorios", [])
        # Reemplazar las primeras N prendas del base con las del estilo
        for i, prenda in enumerate(estilo_prendas):
            if i < len(prendas):
                prendas[i] = prenda          # reemplaza
            else:
                prendas.append(prenda)       # añade si hay hueco
        # Accesorios de estilo reemplazan los del base
        for i, acc in enumerate(estilo_accesorios):
            if i < len(accesorios):
                accesorios[i] = acc
            else:
                accesorios.append(acc)
        justificacion = estilo_conf.get("justificacion", justificacion)

    # Capa 2: ESTACIÓN — añade contexto sin duplicar categorías ya cubiertas
    if estacion_conf:
        estacion_prendas    = estacion_conf.get("prendas", [])
        estacion_accesorios = estacion_conf.get("accesorios", [])
        # Solo añadir si no superamos el límite
        for prenda in estacion_prendas:
            if len(prendas) < MAX_PRENDAS and prenda not in prendas:
                prendas.append(prenda)
        for acc in estacion_accesorios:
            if len(accesorios) < MAX_ACCESORIOS and acc not in accesorios:
                accesorios.append(acc)
        # La justificación de estación complementa si no hay justificación de estilo
        if not estilo_conf:
            justificacion = estacion_conf.get("justificacion", justificacion)

    # Capa 3: CLIMA — solo 1 elemento funcional extra si hay espacio
    if clima_conf:
        clima_prendas    = clima_conf.get("prendas", [])
        clima_accesorios = clima_conf.get("accesorios", [])
        # Solo el primer elemento funcional de clima
        for prenda in clima_prendas[:1]:
            if len(prendas) < MAX_PRENDAS and prenda not in prendas:
                prendas.append(prenda)
        for acc in clima_accesorios[:1]:
            if len(accesorios) < MAX_ACCESORIOS and acc not in accesorios:
                accesorios.append(acc)

    return {
        "prendas":       prendas[:MAX_PRENDAS],
        "accesorios":    accesorios[:MAX_ACCESORIOS],
        "justificacion": justificacion,
    }


def generar_outfit_recomendado(cluster, estacion=None, clima=None, estilo=None):
    info        = outfit_mapping[cluster]
    mood        = info["mood_name"]
    paleta_info = palette_mapping.get(mood, {})
    base        = info["outfit_base"]

    estilo_conf   = info["por_estilo"].get(estilo)            if estilo   else None
    estacion_conf = info["por_estacion"].get(estacion.lower()) if estacion else None
    clima_conf    = info["por_clima"].get(clima.lower())       if clima    else None

    outfit_final = combinar_outfits(
        base,
        estilo_conf=estilo_conf,
        estacion_conf=estacion_conf,
        clima_conf=clima_conf,
    )

    return {
        "mood":                 mood,
        "paleta_colores":       paleta_info.get("colores", []),
        "justificacion_paleta": paleta_info.get("justificacion", ""),
        "outfit_final":         outfit_final,
    }


def predecir_mood_por_titulo(df, titulo, artista, umap_cols, kmeans,
                              estacion=None, clima=None, estilo=None):
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
