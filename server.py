"""
Servidor FastAPI para RAG Histología Neo4j — Fullstack A2UI
============================================================
Wrappea AsistenteHistologiaNeo4j y expone endpoints REST + A2UI.
El módulo ne4j-histo.py se importa sin modificación.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

# ── Importar el módulo principal ─────────────────────────────────────
# ne4j-histo.py tiene guión, así que lo importamos con importlib
import importlib.util

_HISTO_PATH = Path(__file__).parent / "ne4j-histo.py"
spec = importlib.util.spec_from_file_location("ne4j_histo", str(_HISTO_PATH))
ne4j_histo = importlib.util.module_from_spec(spec)

# Prevenir que el módulo ejecute su __main__
_original_argv = sys.argv
sys.argv = ["ne4j-histo.py"]
spec.loader.exec_module(ne4j_histo)
sys.argv = _original_argv

AsistenteHistologiaNeo4j = ne4j_histo.AsistenteHistologiaNeo4j
DIRECTORIO_PDFS = ne4j_histo.DIRECTORIO_PDFS

# ── Estado global ────────────────────────────────────────────────────
asistente: Optional[AsistenteHistologiaNeo4j] = None
_init_complete = False
_init_error: Optional[str] = None


# ── Modelos Pydantic ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    image_filename: Optional[str] = None


class ChatResponse(BaseModel):
    respuesta: str
    estructura_identificada: Optional[str] = None
    imagenes_recuperadas: list = []
    trayectoria: list = []
    imagen_activa: Optional[str] = None


# ── Lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asistente, _init_complete, _init_error
    print("🚀 Iniciando servidor RAG Histología Neo4j + A2UI...")

    try:
        asistente = AsistenteHistologiaNeo4j()
        await asistente.inicializar_componentes()

        print("📚 Leyendo PDFs...")
        asistente.procesar_contenido_base(DIRECTORIO_PDFS)

        print("📋 Extrayendo temario...")
        await asistente.extraer_y_preparar_temario()
        n_temas = len(asistente.extractor_temario.temas) if asistente.extractor_temario else 0
        print(f"   → {n_temas} temas")

        _init_complete = True
        print("✅ Servidor listo")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _init_error = str(e)
        print(f"❌ Error inicializando: {e}")

    yield

    # Shutdown
    if asistente:
        await asistente.cerrar()
    print("👋 Servidor apagado")


# ── App FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Histología Neo4j + A2UI",
    description="Sistema RAG Multimodal de Histología — Fullstack",
    version="4.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Archivos estáticos del cliente
CLIENT_DIR = Path(__file__).parent / "client"


def _check_ready():
    if not _init_complete:
        raise HTTPException(503, detail=_init_error or "Sistema inicializándose...")


# ── Rutas de archivos del frontend ───────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(CLIENT_DIR / "index.html"))


@app.get("/app.js")
async def serve_js():
    return FileResponse(str(CLIENT_DIR / "app.js"), media_type="application/javascript")


@app.get("/style.css")
async def serve_css():
    return FileResponse(str(CLIENT_DIR / "style.css"), media_type="text/css")


# ── API: Estado ──────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    if not _init_complete:
        return {
            "ready": False,
            "error": _init_error,
        }
    return {
        "ready": True,
        "n_temas": len(asistente.extractor_temario.temas) if asistente.extractor_temario else 0,
        "imagen_activa": os.path.basename(asistente.memoria.get_imagen_activa())
            if asistente.memoria and asistente.memoria.get_imagen_activa() else None,
        "turno": asistente.memoria.turno_actual if asistente.memoria else 0,
        "device": asistente.device,
    }


# ── API: Temario ─────────────────────────────────────────────────────
@app.get("/api/temario")
async def get_temario():
    _check_ready()
    temas = asistente.extractor_temario.temas if asistente.extractor_temario else []
    return {"temas": temas, "total": len(temas)}


# ── API: Chat (texto plano) ─────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(req: ChatRequest):
    _check_ready()

    imagen_path = None
    try:
        # Si hay imagen, guardarla como archivo temporal
        if req.image_base64:
            ext = ".png"
            if req.image_filename:
                _, ext = os.path.splitext(req.image_filename)
                if not ext:
                    ext = ".png"
            tmp = tempfile.NamedTemporaryFile(
                suffix=ext, prefix="histo_upload_", delete=False,
                dir=tempfile.gettempdir()
            )
            tmp.write(base64.b64decode(req.image_base64))
            tmp.close()
            imagen_path = tmp.name
            print(f"📷 Imagen guardada: {imagen_path}")

        # Ejecutar consulta RAG
        respuesta = await asistente.consultar(
            consulta_texto=req.query,
            imagen_path=imagen_path,
        )

        # Leer trayectoria
        trayectoria = []
        estructura = None
        imagenes_rec = []
        trayectoria_file = Path(__file__).parent / "trayectoria_neo4j.json"
        if trayectoria_file.exists():
            try:
                with open(trayectoria_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                trayectoria = data.get("trayectoria", [])
                estructura = data.get("estructura_identificada")
                imagenes_rec = data.get("imagenes_recuperadas", [])
            except Exception:
                pass

        img_activa = None
        if asistente.memoria and asistente.memoria.get_imagen_activa():
            img_activa = os.path.basename(asistente.memoria.get_imagen_activa())

        return ChatResponse(
            respuesta=respuesta,
            estructura_identificada=estructura,
            imagenes_recuperadas=[os.path.basename(p) for p in imagenes_rec],
            trayectoria=trayectoria,
            imagen_activa=img_activa,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))
    finally:
        # Limpiar archivo temporal (la imagen ya fue memorizada por el asistente)
        if imagen_path and os.path.exists(imagen_path):
            try:
                os.unlink(imagen_path)
            except Exception:
                pass



# ── API: Limpiar imagen ──────────────────────────────────────────────
@app.post("/api/imagen/limpiar")
async def limpiar_imagen():
    _check_ready()
    if asistente.memoria:
        asistente.memoria.set_imagen(None)
    return {"ok": True, "mensaje": "Imagen activa eliminada"}


# ── Main ─────────────────────────────────────────────────────────────
def main():
    port = int(os.getenv("PORT", "10005"))
    print(f"🌐 Servidor en http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
