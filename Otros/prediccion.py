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
    Combina outfit con prioridad de capas y sistema de categorías:
    - Cada categoría solo puede tener UNA prenda (la de mayor prioridad gana)
    - Prioridad: estilo > estacion > clima > base
    - Accesorios: máximo 3, sin duplicar categoría
    - Justificación: estilo > estacion > base
    """

    # Categorías de prendas para evitar incompatibilidades
    # Orden importa: se evalúa de arriba abajo, la primera coincidencia gana
    CATEGORIAS = [
        # Vestido/mono/conjunto primero — son outfits completos y no deben convivir con top ni pantalón
        ("vestido",  ["vestido", "minivestido", "slip dress", "co-ord ", "conjunto de punto", "set chándal", "set de chándal"]),
        ("mono",     ["mono de", "mono de trabajo", "mono oversize"]),
        # Superior
        ("top",      ["top ", "camiseta", "camisa", "blusa", "crop", "corset", "body de", "bralette", "tirantes", "túnica"]),
        ("jersey",   ["jersey", "sudadera", "hoodie", "forro polar", "cardigan", "cárdigan", "chunky knit", "polo"]),
        # Inferior
        ("pantalon", ["pantalón", "shorts", "falda", "palazzo", "jogger", "wide leg", "pitillo", "cigarette", "chino"]),
        # Capa media
        ("chaqueta", ["chaqueta", "blazer", "bomber", "harrington", "denim"]),
        # Capa externa — solo una (mayor prioridad gana)
        ("abrigo",   ["abrigo", "gabardina", "trench", "puffer", "chubasquero", "impermeable", "anorak", "kimono"]),
        # Calzado
        ("calzado",  ["botas", "botines", "zapatillas", "sneakers", "sandalias", "mocasines", "zapatos"]),
        # Accesorios
        ("sombrero", ["gorro", "gorra", "sombrero", "boina", "beanie", "bucket"]),
        ("bufanda",  ["bufanda", "pañuelo"]),
        ("bolso",    ["bolso", "mochila", "riñonera", "tote"]),
        ("gafas",    ["gafas"]),
        ("joyeria",  ["collar", "cadena", "pendientes", "anillo", "pulsera", "joyería", "ear cuff", "layering", "body chain", "diadema"]),
        ("guantes",  ["guantes", "orejeras"]),
        ("otros_acc",["abanico", "cinturón", "coletero", "calcetines"]),
    ]

    # Categorías que son outfits completos — excluyen top y pantalón
    CATS_OUTFIT_COMPLETO = {"vestido", "mono"}

    def detectar_categoria(prenda: str) -> str:
        p = prenda.lower()
        for cat, keywords in CATEGORIAS:
            if any(kw in p for kw in keywords):
                return cat
        return "otros"

    # Construir outfit por capas, respetando categorías
    # Diccionario: categoria -> prenda (gana la de mayor prioridad)
    prendas_por_cat    = {}
    accesorios_por_cat = {}

    CATS_PRENDAS    = {"top", "jersey", "vestido", "mono", "pantalon", "chaqueta", "abrigo", "calzado"}
    CATS_ACCESORIOS = {"sombrero", "bufanda", "bolso", "gafas", "joyeria", "guantes", "otros_acc"}

    def registrar_item(item, sobreescribir=False):
        cat = detectar_categoria(item)
        if cat in CATS_ACCESORIOS:
            if sobreescribir or cat not in accesorios_por_cat:
                accesorios_por_cat[cat] = item
        else:
            if sobreescribir or cat not in prendas_por_cat:
                prendas_por_cat[cat] = item

    def limpiar_si_outfit_completo():
        """Si hay vestido o mono, elimina top y pantalón independientes."""
        if any(c in prendas_por_cat for c in CATS_OUTFIT_COMPLETO):
            prendas_por_cat.pop("top", None)
            prendas_por_cat.pop("pantalon", None)

    # Orden de prioridad: base (menor) → clima → estacion → estilo (mayor)
    for item in base.get("prendas", []):    registrar_item(item, sobreescribir=False)
    for item in base.get("accesorios", []): registrar_item(item, sobreescribir=False)

    if clima_conf:
        for item in clima_conf.get("prendas", []):    registrar_item(item, sobreescribir=True)
        for item in clima_conf.get("accesorios", []): registrar_item(item, sobreescribir=True)

    if estacion_conf:
        for item in estacion_conf.get("prendas", []):    registrar_item(item, sobreescribir=True)
        for item in estacion_conf.get("accesorios", []): registrar_item(item, sobreescribir=True)

    if estilo_conf:
        for item in estilo_conf.get("prendas", []):    registrar_item(item, sobreescribir=True)
        for item in estilo_conf.get("accesorios", []): registrar_item(item, sobreescribir=True)

    # Si hay vestido o mono, limpiar top y pantalón que puedan haber quedado de capas anteriores
    limpiar_si_outfit_completo()

    # Justificación: estilo > estacion > base
    justificacion = base.get("justificacion", "")
    if estacion_conf and not estilo_conf:
        justificacion = estacion_conf.get("justificacion", justificacion)
    if estilo_conf:
        justificacion = estilo_conf.get("justificacion", justificacion)

    MAX_PRENDAS    = 4
    MAX_ACCESORIOS = 3

    return {
        "prendas":       list(prendas_por_cat.values())[:MAX_PRENDAS],
        "accesorios":    list(accesorios_por_cat.values())[:MAX_ACCESORIOS],
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
