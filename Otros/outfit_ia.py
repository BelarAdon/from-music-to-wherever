import json
import requests


def generar_outfit_con_ia(
    mood: str,
    titulo: str,
    artista: str,
    estacion: str,
    clima: str,
    estilo: str,
    paleta_colores: list,
) -> dict:
    """
    Genera un outfit usando Claude API a partir del mood, contexto y paleta.
    Devuelve un dict con prendas, accesorios y justificacion,
    compatible con el formato de outfit_final de combinar_outfits.
    """

    paleta_str = ", ".join(paleta_colores) if paleta_colores else "sin paleta definida"

    prompt = f"""Eres un estilista de moda experto. Tu tarea es recomendar un outfit coherente y real.

CONTEXTO:
- Canción: "{titulo}" de {artista}
- Mood musical: {mood}
- Estación: {estacion}
- Clima: {clima}
- Estilo personal: {estilo}
- Paleta de colores del mood: {paleta_str}

INSTRUCCIONES:
- Recomienda exactamente 3-4 prendas que formen un outfit coherente y ponible
- Recomienda exactamente 2-3 accesorios que complementen el outfit
- Las prendas deben tener sentido juntas (no mezclar prendas incompatibles)
- Usa los colores de la paleta como guía, no como obligación estricta
- Adapta al clima y estación de forma práctica
- Escribe una justificación de 2 frases máximo explicando la elección

Responde SOLO con este JSON, sin texto adicional ni backticks:
{{
  "prendas": ["prenda1", "prenda2", "prenda3"],
  "accesorios": ["accesorio1", "accesorio2"],
  "justificacion": "Explicación breve del outfit."
}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        raw = data["content"][0]["text"].strip()
        # Limpiar posibles backticks residuales
        raw = raw.replace("```json", "").replace("```", "").strip()
        outfit = json.loads(raw)

        # Validar estructura mínima
        if not all(k in outfit for k in ["prendas", "accesorios", "justificacion"]):
            raise ValueError("Respuesta incompleta de la API")

        return {
            "prendas":       outfit["prendas"][:4],
            "accesorios":    outfit["accesorios"][:3],
            "justificacion": outfit["justificacion"],
            "fuente":        "ia",
        }

    except Exception as e:
        return {"error": f"Error al generar outfit con IA: {e}"}
