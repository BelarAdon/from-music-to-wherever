from Otros.cargar_modelos import scaler, kmeans, umap_model, feature_cols
from Otros.preprocesador import preprocesar_features
from Otros.outfit_mapping import outfit_mapping
from Otros.palette_mapping import palette_mapping


def predecir_mood(features_dict, feature_cols, scaler, umap_model, kmeans):
    emb = preprocesar_features(features_dict)
    cluster = int(kmeans.predict(emb)[0])
    mood = outfit_mapping[cluster]["mood_name"]
    return cluster, mood

def generar_outfit_recomendado(cluster, estacion=None, clima=None, estilo=None):
    info = outfit_mapping[cluster]
    mood = info["mood_name"]
    paleta_info = palette_mapping.get(mood, {})
    paleta = paleta_info.get("colores",[])
    justificacion_paleta = paleta_info.get("justificacion","")

    base = info["outfit_base"]

    estilo_conf = info["por_estilo"].get(estilo) if estilo else None
    estacion_conf = info["por_estacion"].get(estacion.lower()) if estacion else None
    clima_conf = info["por_clima"].get(clima.lower()) if clima else None

    outfit_final = combinar_outfits(
        base,
        estilo_conf=estilo_conf,
        estacion_conf=estacion_conf,
        clima_conf=clima_conf
    )

    return {
        "mood": mood,
        "paleta_colores": paleta,
        "justificacion_paleta": justificacion_paleta,
        "outfit_final": outfit_final
    }



import pandas as pd

def buscar_cancion(df, titulo, artista=None):
    titulo = titulo.lower()

    df_filtrado = df[df['track_name'].str.lower().str.contains(titulo, na=False)]

    if artista:
        artista = artista.lower()
        df_filtrado = df_filtrado[df_filtrado['track_artist'].str.lower().str.contains(artista, na=False)]

    if df_filtrado.empty:
        return None

    return df_filtrado.iloc[0]


def predecir_mood_por_titulo(
    df,
    titulo,
    artista,
    feature_cols,
    scaler,
    umap_model,
    kmeans,
    estacion=None,
    clima=None,
    estilo=None
):

    fila = buscar_cancion(df, titulo, artista)
    if fila is None:
        return {"error": "Canción no encontrada"}


    features_dict = fila[feature_cols].to_dict()

    cluster, mood = predecir_mood(
        features_dict,
        feature_cols,
        scaler,
        umap_model,
        kmeans
    )
    outfit = generar_outfit_recomendado(
        cluster,
        estacion=estacion,
        clima=clima,
        estilo=estilo
    )

    return {
        "title": fila["track_name"],
        "artist": fila.get("track_artist", "Desconocido"),
        "mood": mood,
        "cluster": cluster,
        "outfit": outfit
    }




def combinar_outfits(base, estilo_conf=None, estacion_conf=None, clima_conf=None):
    prendas = set(base.get("prendas", []))
    accesorios = set(base.get("accesorios", []))
    justificaciones = [base.get("justificacion", "")]

    # Mezcla estilo personal
    if estilo_conf:
        prendas.update(estilo_conf.get("prendas", []))
        accesorios.update(estilo_conf.get("accesorios", []))
        justificaciones.append(estilo_conf.get("justificacion", ""))

    # Mezcla estación
    if estacion_conf:
        prendas.update(estacion_conf.get("prendas", []))
        accesorios.update(estacion_conf.get("accesorios", []))
        justificaciones.append(estacion_conf.get("justificacion", ""))

    # Mezcla clima
    if clima_conf:
        prendas.update(clima_conf.get("prendas", []))
        accesorios.update(clima_conf.get("accesorios", []))
        justificaciones.append(clima_conf.get("justificacion", ""))

    return {
        "prendas": list(prendas),
        "accesorios": list(accesorios),
        "justificacion": " ".join(justificaciones)
    }

