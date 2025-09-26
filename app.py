import os, io, re, json, requests
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# —— librerías para el modo demo (extractivo)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

load_dotenv()
st.set_page_config(page_title="PDF -> Resumen configurable", page_icon="📄", layout="wide")

# descargar tokenizador para sumy/nltk (idempotente)
try:
    nltk.download("punkt", quiet=True)
except Exception:
    pass

# ============================= Utilidades PDF =============================
def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n\n".join(pages_text)

def chunk_text(text: str, max_chars: int = 7000, overlap: int = 500) -> List[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = max(end - overlap, 0)
    return chunks

# ============================= Configuración de secciones =============================
DEFAULT_OBJECTIVE = (
    "Generar un resumen ejecutivo claro y estructurado del documento, "
    "enfocado en decisiones y hallazgos clave."
)

DEFAULT_SECTIONS = [
    {"id": "titulo", "label": "Título Principal",
     "instruction": "Encabeza con la acción principal o idea-fuerza del documento.", "required": True},
    {"id": "contexto", "label": "Contexto",
     "instruction": "Explica brevemente el contexto, alcance y propósito.", "required": True},
    {"id": "hallazgos", "label": "Hallazgos/Conclusiones",
     "instruction": "Enumera los hallazgos clave y sus implicancias.", "required": True},
    {"id": "afectacion", "label": "Afectación/Impacto (omitir si no aplica)",
     "instruction": "Solo incluir si el documento contiene datos claros de afectación/impacto; si no hay evidencia, omitir por completo esta sección.",
     "required": False, "omit_if_empty": True},
    {"id": "recomendaciones", "label": "Recomendaciones/Cursos de acción",
     "instruction": "Lista recomendaciones accionables priorizadas (si existen).", "required": False},
    {"id": "fuentes", "label": "Fuentes/Referencias (opcional)",
     "instruction": "Cita brevemente secciones/páginas relevantes del PDF.", "required": False},
]

# ============================= Motores de resumen =============================
def summarize_demo_extractive(text: str, sentences: int = 8) -> str:
    """Resumen extractivo (sin API) con tolerancia a fallos.
    1) Intenta usar Sumy + NLTK (TextRank).
    2) Si falla, usa un respaldo simple por oraciones.
    """
    try:
        # Capar longitud para evitar caídas con textos enormes
        text = (text or "").strip()
        if not text:
            return ""
        if len(text) > 20000:
            text = text[:20000]

        # Intento 1: Sumy + NLTK (TextRank)
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            parser = PlaintextParser.from_string(text, Tokenizer("spanish"))
            summarizer = TextRankSummarizer()
            sents = summarizer(parser.document, sentences)
            out = [str(s) for s in sents if str(s).strip()]
            if out:
                return "• " + "\n• ".join(out)
        except Exception:
            pass

        # Respaldo: segmentar por oraciones y puntuar
        import re
        # Separar por punto/salto de línea
        raw_sents = re.split(r"(?<=[.!?])\s+|\n+", text)
        # Limpieza básica
        sents = [s.strip() for s in raw_sents if len(s.strip()) > 0]

        if not sents:
            return ""

        # Puntuar: preferir oraciones con números, %, mayúsculas iniciales, o palabras clave
        keywords = ("alerta", "afectación", "impacto", "personas", "daños",
                    "declaración", "nivel", "recomendación", "curso de acción",
                    "evento", "riesgo", "región", "comuna", "provincia", "reporte")
        def score(sent: str) -> int:
            sc = 0
            if re.search(r"\d", sent): sc += 2
            if "%" in sent: sc += 2
            if any(k in sent.lower() for k in keywords): sc += 3
            if len(sent) > 80: sc += 1  # algo de preferencia por oraciones más informativas
            return sc

        ranked = sorted(sents, key=score, reverse=True)[:max(3, sentences)]
        return "• " + "\n• ".join(ranked)

    except Exception as e:
        # Último recurso: devolver primeros N fragmentos
        short = text[:1200]
        return "• " + "\n• ".join([s.strip() for s in short.split("\n") if s.strip()][:sentences])

def summarize_hf_bart(text: str, hf_token: str, max_len: int = 300) -> str:
    """Resumen con Hugging Face Inference API (requiere token gratuito)."""
    if not hf_token:
        raise RuntimeError("Falta HF_TOKEN en Secrets.")
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text, "parameters": {"max_length": max_len}}
    r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    # Respuesta típica: [{"summary_text": "..."}]
    if isinstance(data, list) and data and "summary_text" in data[0]:
        return data[0]["summary_text"]
    return json.dumps(data)  # por si el modelo devuelve otra forma

# OpenAI queda como opcional (si después activas billing):
def summarize_openai(text: str, model: str, api_key: str) -> str:
    from openai import OpenAI
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()
    prompt = (
        "Resume el siguiente texto en español en 8-10 viñetas claras, "
        "sin inventar información y manteniendo datos clave:\n\n" + text
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ============================= UI =============================
st.title("📄 PDF → Resumen configurable")

with st.sidebar:
    st.header("⚙️ Configuración")
    backend = st.selectbox(
        "Motor de resumen",
        ["Modo demo (sin API)", "Hugging Face (gratis con token)", "OpenAI (API de pago)"],
        index=0,
        help="Elige 'Modo demo' si no quieres usar ninguna API."
    )

    # Modelos solo para OpenAI (si algún día lo activas)
    model = st.text_input("Modelo (solo OpenAI)", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature (solo OpenAI)", 0.0, 1.0, 0.2, 0.05)
    max_chars = st.number_input("Tamaño de trozo (chars)", min_value=1000, max_value=12000, value=4000, step=500)
    overlap = st.number_input("Solapamiento (chars)", min_value=0, max_value=2000, value=200, step=50)

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
            "id": s["id"], "label": label, "instruction": instruction,
            "required": required, "omit_if_empty": omit_if_empty,
        })

    st.divider()
    st.caption("Credenciales (solo si usas API)")
    api_key = st.secrets.get("OPENAI_API_KEY", "") or st.text_input("OPENAI_API_KEY (OpenAI)", type="password", value="")
    hf_token = st.secrets.get("HF_TOKEN", "") or st.text_input("HF_TOKEN (Hugging Face)", type="password", value="")

uploaded = st.file_uploader("Sube un PDF", type=["pdf"])

# Diagnóstico simple
with st.expander("🔎 Diagnóstico rápido", expanded=False):
    st.write({"backend": backend, "OPENAI_API_KEY?": bool(api_key), "HF_TOKEN?": bool(hf_token)})
    if st.button("🧪 Probar motor seleccionado"):
        try:
            if backend == "Modo demo (sin API)":
                st.success("No se necesita API. ✅")
            elif backend == "Hugging Face (gratis con token)":
                if not hf_token:
                    st.error("Falta HF_TOKEN.")
                else:
                    out = summarize_hf_bart("Texto de prueba para resumen.", hf_token)
                    st.success("Conexión HF OK ✅")
                    st.code(out, language="text")
            else:  # OpenAI
                if not api_key:
                    st.error("Falta OPENAI_API_KEY.")
                else:
                    out = summarize_openai("Texto de prueba para resumen.", model, api_key)
                    st.success("Conexión OpenAI OK ✅")
                    st.code(out, language="text")
        except Exception as e:
            st.error("❌ Falló la prueba.")
            st.exception(e)

# Flujo principal
if uploaded is not None:
    try:
        data = uploaded.getvalue()
        reader = PdfReader(io.BytesIO(data))
        text = extract_text_from_pdf(data)
        st.info(f"Páginas detectadas: {len(reader.pages)} | Longitud texto: {len(text)} chars")

        if st.button("▶️ Generar resumen"):
            with st.spinner("Procesando…"):
                chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

                partials = []
                for i, ch in enumerate(chunks):
                    if backend == "Modo demo (sin API)":
                        out = summarize_demo_extractive(ch, sentences=8)
                    elif backend == "Hugging Face (gratis con token)":
                        out = summarize_hf_bart(ch, hf_token, max_len=300)
                    else:  # OpenAI
                        out = summarize_openai(ch, model, api_key)
                    partials.append(out)
                    st.write(f"Trozo {i+1}/{len(chunks)} procesado.")

                # “Agregación” sencilla (concatenar y cortar)
                final = "\n\n".join(partials)
                st.subheader("Resumen final")
                st.code(final, language="markdown")

                st.download_button("💾 Descargar TXT", "resumen.txt", "text/plain", final.encode("utf-8"))

    except Exception as e:
        st.error("❌ Error durante el procesamiento.")
        st.exception(e)
else:
    st.warning("Sube un PDF para comenzar.")
