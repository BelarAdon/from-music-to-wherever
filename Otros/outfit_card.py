import streamlit.components.v1 as components

# Mapeo de nombres de color a hex
COLOR_HEX = {
    # Concierto / Euforia
    "negro":           "#0D0D0D",
    "rojo eléctrico":  "#FF1A1A",
    "plateado":        "#C0C0C0",
    "morado neón":     "#BF00FF",
    # Intenso / Dramático
    "negro profundo":  "#111111",
    "borgoña":         "#6E0B2A",
    "azul noche":      "#0A1045",
    "gris carbón":     "#3B3B3B",
    # Instrumental / Concentración
    "beige":           "#E8DCCB",
    "crema":           "#FFF8E7",
    "verde oliva":     "#6B7C45",
    "azul suave":      "#A8C4D4",
    # Chill Groove / Urbano suave
    "camel":           "#C19A6B",
    "khaki":           "#C3B091",
    "terracota":       "#C46A45",
    "negro suave":     "#2B2B2B",
    # Happy / Buen Rollo
    "amarillo pastel": "#FFE566",
    "coral":           "#FF6B6B",
    "azul cielo":      "#87CEEB",
    "verde menta":     "#A8E6CF",
    # Fiesta / Subidón
    "fucsia":          "#FF00AA",
    "negro brillante": "#0A0A0A",
    "azul eléctrico":  "#0066FF",
    # Triste / Melancólico
    "azul grisáceo":   "#7B9BAD",
    "lavanda":         "#C4B5D4",
    "gris suave":      "#B0B0B0",
    "negro apagado":   "#3A3A3A",
}

# Emoji por mood
MOOD_EMOJI = {
    "Concierto / Euforia":           "🎸",
    "Intenso / Dramático":           "🎭",
    "Instrumental / Concentración":  "🎹",
    "Chill groove / Urbano suave":   "🌿",
    "Chill Groove / Urbano suave":   "🌿",
    "Happy / Buen Rollo":            "☀️",
    "Fiesta / Subidón":              "🪩",
    "Triste / Melancólico":          "🌧️",
}

def render_outfit_card(res: dict, height: int = 520):
    """Renderiza la tarjeta de outfit en Streamlit."""
    outfit       = res["outfit"]
    outfit_final = outfit["outfit_final"]
    mood         = res["mood"]
    title        = res["title"]
    artist       = res["artist"]
    prendas      = outfit_final["prendas"]
    accesorios   = outfit_final["accesorios"]
    justificacion= outfit_final["justificacion"]
    colores      = outfit.get("paleta_colores", [])
    just_paleta  = outfit.get("justificacion_paleta", "")
    emoji        = MOOD_EMOJI.get(mood, "🎵")

    # Chips de prendas
    def chips(items, color):
        return "".join(
            f'<span style="display:inline-block;background:{color}18;color:{color};'
            f'border:1px solid {color}40;border-radius:20px;padding:4px 12px;'
            f'font-size:13px;margin:3px 4px 3px 0;">{item}</span>'
            for item in items
        )

    # Círculos de color
    color_circles = ""
    for nombre in colores:
        hex_val = COLOR_HEX.get(nombre.lower(), "#888888")
        is_light = nombre in ["crema", "beige", "amarillo pastel", "azul cielo", "verde menta",
                               "lavanda", "gris suave", "azul suave", "khaki", "plateado"]
        border = f"border:1.5px solid #00000020;" if is_light else ""
        color_circles += (
            f'<div title="{nombre}" style="width:32px;height:32px;border-radius:50%;'
            f'background:{hex_val};{border}flex-shrink:0;"></div>'
        )

    # Color de acento según mood
    MOOD_ACCENT = {
        "Concierto / Euforia":          ("#FF1A1A", "#2a0000"),
        "Intenso / Dramático":          ("#6E0B2A", "#1a0008"),
        "Instrumental / Concentración": ("#6B7C45", "#1a1f0f"),
        "Chill groove / Urbano suave":  ("#C46A45", "#2a1610"),
        "Chill Groove / Urbano suave":  ("#C46A45", "#2a1610"),
        "Happy / Buen Rollo":           ("#FF6B6B", "#2a0d0d"),
        "Fiesta / Subidón":             ("#FF00AA", "#2a0020"),
        "Triste / Melancólico":         ("#7B9BAD", "#101820"),
    }
    accent, dark = MOOD_ACCENT.get(mood, ("#888888", "#1a1a1a"))

    prendas_html    = chips(prendas,    accent)
    accesorios_html = chips(accesorios, "#888888")

    html = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  .card {{
    font-family: 'Inter', sans-serif;
    background: #111;
    border-radius: 16px;
    overflow: hidden;
    color: #fff;
    max-width: 680px;
  }}
  .card-header {{
    background: linear-gradient(135deg, {dark} 0%, #111 100%);
    border-bottom: 1px solid {accent}30;
    padding: 20px 24px 16px;
  }}
  .mood-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {accent}20; color: {accent};
    border: 1px solid {accent}40;
    border-radius: 20px; padding: 4px 12px;
    font-size: 12px; font-weight: 500; margin-bottom: 10px;
  }}
  .song-title {{
    font-size: 18px; font-weight: 600; color: #fff;
    margin: 0 0 2px; line-height: 1.3;
  }}
  .song-artist {{
    font-size: 13px; color: #888; margin: 0;
  }}
  .card-body {{ padding: 20px 24px; }}
  .section-label {{
    font-size: 10px; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #555; margin-bottom: 8px;
  }}
  .section {{ margin-bottom: 20px; }}
  .palette-row {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 8px;
  }}
  .just-text {{
    font-size: 12px; color: #666; line-height: 1.5; margin: 0;
  }}
  .divider {{
    border: none; border-top: 1px solid #222; margin: 4px 0 20px;
  }}
</style>

<div class="card">
  <div class="card-header">
    <div class="mood-badge">{emoji} {mood}</div>
    <p class="song-title">{title}</p>
    <p class="song-artist">{artist}</p>
  </div>
  <div class="card-body">

    <div class="section">
      <div class="section-label">Prendas</div>
      <div>{prendas_html}</div>
    </div>

    <div class="section">
      <div class="section-label">Accesorios</div>
      <div>{accesorios_html}</div>
    </div>

    <hr class="divider">

    <div class="section">
      <div class="section-label">Paleta de color</div>
      <div class="palette-row">
        {color_circles}
      </div>
      <p class="just-text">{just_paleta}</p>
    </div>

    <div class="section" style="margin-bottom:0">
      <div class="section-label">Estilo</div>
      <p class="just-text">{justificacion}</p>
    </div>

  </div>
</div>
"""
    components.html(html, height=height, scrolling=False)
