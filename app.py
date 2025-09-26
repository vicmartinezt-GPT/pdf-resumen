import os
import math
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

load_dotenv()

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n\n".join(pages_text)

def chunk_text(text: str, max_chars: int = 7000, overlap: int = 500) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

DEFAULT_OBJECTIVE = (
    "Generar un resumen ejecutivo claro y estructurado del documento, "
    "enfocado en decisiones y hallazgos clave."
)

DEFAULT_SECTIONS = [
    {
        "id": "titulo",
        "label": "Título Principal",
        "instruction": "Encabeza con la acción principal o idea-fuerza del documento.",
        "required": True,
    },
    {
        "id": "contexto",
        "label": "Contexto",
        "instruction": "Explica brevemente el contexto, alcance y propósito.",
        "required": True,
    },
    {
        "id": "hallazgos",
        "label": "Hallazgos/Conclusiones",
        "instruction": "Enumera los hallazgos clave y sus implicancias.",
        "required": True,
    },
    {
        "id": "afectacion",
        "label": "Afectación/Impacto (omitir si no aplica)",
        "instruction": (
            "Solo incluir si el documento contiene datos claros de afectación/impacto; "
            "si no hay evidencia, omitir por completo esta sección."
        ),
        "required": False,
        "omit_if_empty": True,
    },
    {
        "id": "recomendaciones",
        "label": "Recomendaciones/Cursos de acción",
        "instruction": "Lista recomendaciones accionables priorizadas (si existen).",
        "required": False,
    },
    {
        "id": "fuentes",
        "label": "Fuentes/Referencias (opcional)",
        "instruction": "Cita brevemente secciones/páginas relevantes del PDF.",
        "required": False,
    },
]

AGGREGATION_SYSTEM_PROMPT = (
    "Eres un analista experto en síntesis documental. Recibes resúmenes parciales "
    "(por trozos) y debes consolidarlos en un único resumen final coherente, sin "
    "repeticiones, manteniendo trazabilidad ligera a páginas cuando sea posible."
)

CHUNK_SYSTEM_PROMPT = (
    "Eres un analista experto. Resume el contenido del trozo dado en relación con el objetivo y las secciones definidas. "
    "Respeta las reglas de omitir secciones cuando no haya evidencia. No inventes información."
)

def call_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def build_chunk_user_prompt(objetivo: str, secciones: List[Dict[str, Any]], chunk: str, idx: int, total: int) -> str:
    template = [
        f"OBJETIVO: {objetivo}",
        "SECCIONES (con reglas):",
    ]
    for s in secciones:
        rule = " [OMITIR SI VACÍO]" if s.get("omit_if_empty") else (" [REQUERIDA]" if s.get("required") else "")
        template.append(f"- {s['label']}: {s['instruction']}{rule}")
    template.append("")
    template.append(f"TROZO {idx+1}/{total} (no inventes, no extrapoles sin evidencia):")
    template.append(chunk)
    template.append("")
    template.append("DEVUELVE UN JSON con claves igual al id de cada sección y valores textuales limpios. \n"
                    "Incluye solo secciones con contenido; si una sección no aplica o está vacía, omítela del JSON.")
    return "\n".join(template)

def build_aggregation_user_prompt(objetivo: str, secciones: List[Dict[str, Any]], partial_json_list: List[str]) -> str:
    template = [
        f"OBJETIVO: {objetivo}",
        "SECCIONES TOTALES (estructura final):",
    ]
    for s in secciones:
        rule = " [OMITIR SI VACÍO]" if s.get("omit_if_empty") else (" [REQUERIDA]" if s.get("required") else "")
        template.append(f"- {s['id']}: {s['label']} — {rule}")
    template.append("")
    template.append("A continuación tienes una lista de resúmenes parciales en formato JSON (uno por trozo). \n"
                    "Fusiónalos en un único resumen final coherente, sin repeticiones, siguiendo el orden de secciones.\n"
                    "No inventes información no presente en los parciales.")
    template.append("\nRESÚMENES PARCIALES:\n" + "\n\n".join(partial_json_list))
    template.append("")
    template.append("DEVUELVE SOLO UN JSON FINAL con las claves de sección y valores textuales (Markdown permitido).")
    return "\n".join(template)

st.set_page_config(page_title="PDF -> Resumen configurable", page_icon="📄", layout="wide")
st.title("📄 PDF → Resumen configurable")

with st.sidebar:
    st.header("⚙️ Configuración")
    model = st.text_input("Modelo (chat.completions)", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_chars = st.number_input("Tamaño de trozo (chars)", min_value=2000, max_value=12000, value=7000, step=500)
    overlap = st.number_input("Solapamiento (chars)", min_value=0, max_value=2000, value=500, step=50)

    st.divider()
    st.caption("Define objetivo y secciones del resumen")
    objetivo = st.text_area("Objetivo del resumen", value=DEFAULT_OBJECTIVE, height=80)

    st.subheader("Secciones")
    edited_sections: List[Dict[str, Any]] = []
    for s in DEFAULT_SECTIONS:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            label = st.text_input(f"Etiqueta — {s['id']}", value=s["label"], key=f"label_{s['id']}")
            instruction = st.text_area(f"Instrucción — {s['id']}", value=s["instruction"], key=f"instr_{s['id']}")
        with col2:
            required = st.checkbox("Requerida", value=s.get("required", False), key=f"req_{s['id']}")
        with col3:
            omit_if_empty = st.checkbox("Omitir si vacío", value=s.get("omit_if_empty", False), key=f"omit_{s['id']}")
        edited_sections.append({
            "id": s["id"],
            "label": label,
            "instruction": instruction,
            "required": required,
            "omit_if_empty": omit_if_empty,
        })

    st.divider()
    st.caption("Claves de API")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))

uploaded = st.file_uploader("Sube un PDF", type=["pdf"]) 

if uploaded is not None and api_key:
    text = extract_text_from_pdf(uploaded)
    st.info(f"Páginas detectadas: {len(PdfReader(uploaded).pages)} | Longitud texto: {len(text)} chars")

    if st.button("▶️ Generar resumen"):
        with st.spinner("Procesando…"):
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
            client = OpenAI(api_key=api_key)

            partials = []
            for i, ch in enumerate(chunks):
                uprompt = build_chunk_user_prompt(objetivo, edited_sections, ch, i, len(chunks))
                out = call_llm(client, model, CHUNK_SYSTEM_PROMPT, uprompt)
                partials.append(out)
                st.write(f"Trozo {i+1}/{len(chunks)} procesado.")

            agg_prompt = build_aggregation_user_prompt(objetivo, edited_sections, partials)
            final_json = call_llm(client, model, AGGREGATION_SYSTEM_PROMPT, agg_prompt)

            st.subheader("Resumen final (JSON)")
            st.code(final_json, language="json")

            st.download_button(
                label="💾 Descargar JSON",
                file_name="resumen.json",
                mime="application/json",
                data=final_json.encode("utf-8"),
            )
else:
    st.warning("Sube un PDF y define tu OPENAI_API_KEY en la barra lateral para comenzar.")
