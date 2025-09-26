
# PDF → Resumen configurable (Streamlit)

## Uso local (opcional)
1. Instala Python 3.10+
2. En la terminal:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Crea un archivo `.env` con `OPENAI_API_KEY=...`

## Despliegue sin instalar nada (Streamlit Community Cloud)
1. Crea una cuenta en GitHub y un repositorio nuevo.
2. Sube `app.py` y `requirements.txt`.
3. Ve a https://streamlit.io/ -> **Sign in** -> **New app** -> conecta tu repo y elige `app.py`.
4. En **Manage app** -> **Secrets** agrega:
   ```
   OPENAI_API_KEY = "tu_clave_aqui"
   ```
5. Pulsa **Deploy**. Carga un PDF y usa **Generar resumen**.
