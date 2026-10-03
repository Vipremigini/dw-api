"""
DeepWriting Headless REST API
FastAPI service exposing handwriting generation for web services, pen plotters, and mobile apps.
Designed for Hugging Face Spaces (Docker / Free Tier).
"""

import asyncio
import io
import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import deepwriting

# Global writer instance & execution lock for thread-safe inference
writer: Optional[deepwriting.DeepWriter] = None
model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads model into RAM at startup, unloads on shutdown."""
    global writer
    print("[DeepWriting API] Initializing model...")
    writer = deepwriting.load_model(verbose=True)
    print("[DeepWriting API] Model loaded successfully and ready to serve requests.")
    yield
    print("[DeepWriting API] Shutting down...")
    if writer is not None:
        writer.unload()
        writer = None
    print("[DeepWriting API] Shutdown complete.")


app = FastAPI(
    title="DeepWriting Headless API",
    description=(
        "Headless REST API for **DeepWriting: Making Digital Ink Editable via Deep Generative Modeling** (CHI '18).\n\n"
        "Generates true online pen stroke trajectories from text. Supports export to:\n"
        "- **Single-line centerline SVG** (for pen plotters, CNC, laser engravers, AxiDraw)\n"
        "- **High-resolution PNG**\n"
        "- **Raw trajectory JSON** (offsets & continuous line segments)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external frontends / web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Models ---

class SynthesizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text to synthesize into handwriting",
        json_schema_extra={"example": "DeepWriting generates realistic digital ink."},
    )
    style: Optional[int] = Field(
        107,
        description="Reference handwriting style sample index (0 to 704). Pass null or -1 for random unbiased style.",
        json_schema_extra={"example": 107},
    )
    format: str = Field(
        "svg",
        description="Output format: 'svg', 'png', or 'json'",
        json_schema_extra={"example": "svg"},
    )
    beautify: bool = Field(
        True,
        description="Apply mean-sampling filter for smoothed, neat handwriting",
    )
    stroke_width: float = Field(
        2.0,
        ge=0.1,
        le=20.0,
        description="Stroke line width for SVG or PNG",
    )
    color: str = Field(
        "#000000",
        description="Ink color in hex (#000000) or CSS color name",
    )
    background: Optional[str] = Field(
        None,
        description="Background fill color. Defaults to null (transparent for SVG, white for PNG).",
    )
    dpi: int = Field(
        200,
        ge=72,
        le=600,
        description="Resolution DPI for PNG rendering",
    )


class SynthesizeJsonResponse(BaseModel):
    text: str
    style_id: Optional[int]
    beautified: bool
    stroke_count: int
    strokes: List[List[float]] = Field(
        ..., description="Raw stroke offsets shape (N, 3): [dx, dy, pen_up]"
    )
    lines: List[List[List[float]]] = Field(
        ..., description="List of continuous unbroken stroke segments [[(x, y), ...], ...]"
    )
    svg: str = Field(..., description="Single-line centerline SVG XML string")


# --- Helper Function ---

def _run_synthesis(
    text: str,
    style: Optional[int],
    beautify: bool,
) -> deepwriting.SynthesisResult:
    """Invokes synthesis on the global writer instance."""
    if writer is None or not writer.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently loading or unavailable.",
        )
    try:
        return writer.synthesize(text=text, style=style, beautify=beautify)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis error: {str(e)}",
        )


# --- Endpoints ---

@app.get("/", tags=["Info"])
def root_info():
    """Service metadata, health status, and links to documentation."""
    is_ready = writer is not None and writer.is_loaded
    return {
        "service": "DeepWriting Headless REST API",
        "status": "ready" if is_ready else "initializing",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "endpoints": {
            "health": "/health",
            "styles": "/v1/styles",
            "synthesize_post": "POST /v1/synthesize",
            "synthesize_svg_get": "GET /v1/synthesize/svg?text=...&style=107",
            "synthesize_png_get": "GET /v1/synthesize/png?text=...&style=107",
        },
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health probe for container orchestrators (Hugging Face / Kubernetes)."""
    is_ready = writer is not None and writer.is_loaded
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "initializing", "model_loaded": False},
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/v1/styles", tags=["Styles"])
def list_styles():
    """Returns available handwriting style indices in the loaded dataset."""
    if writer is None or not writer.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently loading.",
        )
    available_styles = writer.list_available_styles()
    return {
        "total_styles": len(available_styles),
        "styles": available_styles,
        "recommended": [107, 226, 696, 50, 300],
        "random_style_option": -1,
    }


@app.post(
    "/v1/synthesize",
    response_model=Optional[SynthesizeJsonResponse],
    tags=["Synthesis"],
    summary="Synthesize handwriting (POST)",
)
async def synthesize_post(req: SynthesizeRequest):
    """
    Synthesize handwriting from text.
    Returns SVG XML, binary PNG, or structured JSON trajectory data based on `format`.
    """
    async with model_lock:
        res = await asyncio.to_thread(_run_synthesis, req.text, req.style, req.beautify)

    fmt = req.format.lower().strip()
    if fmt == "svg":
        svg_content = res.to_svg_string(
            stroke_width=req.stroke_width,
            color=req.color,
            background=req.background,
        )
        return Response(content=svg_content, media_type="image/svg+xml")

    elif fmt == "png":
        buf = io.BytesIO()
        res.save_png(
            buf,
            linewidth=req.stroke_width,
            color=req.color,
            dpi=req.dpi,
        )
        return Response(content=buf.getvalue(), media_type="image/png")

    elif fmt == "json":
        # Format continuous lines into JSON-serializable list
        lines_json = [line.tolist() for line in res.lines]
        svg_str = res.to_svg_string(
            stroke_width=req.stroke_width,
            color=req.color,
            background=req.background,
        )
        return SynthesizeJsonResponse(
            text=res.text,
            style_id=res.style_id,
            beautified=res.beautified,
            stroke_count=len(res),
            strokes=res.strokes.tolist(),
            lines=lines_json,
            svg=svg_str,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{req.format}'. Choose from 'svg', 'png', or 'json'.",
        )


@app.get(
    "/v1/synthesize/svg",
    tags=["Synthesis"],
    summary="Direct GET SVG synthesis (for <img> tags & browsers)",
)
async def synthesize_svg_get(
    text: str = Query(..., min_length=1, max_length=500, description="Text to synthesize"),
    style: Optional[int] = Query(107, description="Style index (or -1 for random)"),
    stroke_width: float = Query(2.0, ge=0.1, le=20.0, description="Stroke line width"),
    color: str = Query("#000000", description="Ink color"),
    background: Optional[str] = Query(None, description="Background color"),
    beautify: bool = Query(True, description="Enable stroke smoothing"),
):
    """Generates and returns an SVG vector graphic directly via GET."""
    async with model_lock:
        res = await asyncio.to_thread(_run_synthesis, text, style, beautify)

    svg_content = res.to_svg_string(
        stroke_width=stroke_width,
        color=color,
        background=background,
    )
    return Response(
        content=svg_content,
        media_type="image/svg+xml",
        headers={"Content-Disposition": 'inline; filename="handwriting.svg"'},
    )


@app.get(
    "/v1/synthesize/png",
    tags=["Synthesis"],
    summary="Direct GET PNG synthesis (for <img> tags & browsers)",
)
async def synthesize_png_get(
    text: str = Query(..., min_length=1, max_length=500, description="Text to synthesize"),
    style: Optional[int] = Query(107, description="Style index (or -1 for random)"),
    linewidth: float = Query(2.0, ge=0.1, le=20.0, description="Stroke line width"),
    color: str = Query("#000000", description="Ink color"),
    dpi: int = Query(200, ge=72, le=600, description="Resolution DPI"),
    beautify: bool = Query(True, description="Enable stroke smoothing"),
):
    """Generates and returns a PNG raster image directly via GET."""
    async with model_lock:
        res = await asyncio.to_thread(_run_synthesis, text, style, beautify)

    buf = io.BytesIO()
    res.save_png(buf, linewidth=linewidth, color=color, dpi=dpi)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="handwriting.png"'},
    )


# Optional Gradio UI (enabled only when ENABLE_GRADIO=1, e.g. for Hugging Face Spaces)
if os.environ.get("ENABLE_GRADIO", "0") == "1":
    try:
        import gradio as gr

        def gradio_synthesize(text: str, style_id: int, beautify: bool):
            """UI helper to synthesize handwriting and display as SVG HTML."""
            if not text or not text.strip():
                return "<p style='color: red;'>Please enter text to synthesize.</p>", ""
            style_val = int(style_id) if style_id != -1 else None
            res = _run_synthesis(text=text, style=style_val, beautify=beautify)
            svg_html = res.to_svg_string(stroke_width=2.0, color="#111827")
            preview = (
                f'<div style="background: white; padding: 20px; border-radius: 8px; '
                f'border: 1px solid #e5e7eb; display: flex; justify-content: center; '
                f'align-items: center; overflow-x: auto; min-height: 120px;">{svg_html}</div>'
            )
            info = f"✓ Generated **{len(res)} stroke points** across **{len(res.lines)} continuous lines**."
            return preview, info

        with gr.Blocks(title="DeepWriting API") as demo:
            gr.Markdown("# ✍️ DeepWriting: Handwriting Generation API")
            gr.Markdown(
                "A headless REST API hosted on **Hugging Face Spaces (Free Tier)**.\n\n"
                "🔗 **[Interactive Swagger API Docs](/docs)** &nbsp;|&nbsp; "
                "**[OpenAPI JSON Schema](/openapi.json)** &nbsp;|&nbsp; "
                "**[Health Probe](/health)**"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Input Text",
                        value="DeepWriting on Hugging Face Spaces",
                        lines=2,
                        placeholder="Enter words to synthesize...",
                    )
                    with gr.Row():
                        style_input = gr.Dropdown(
                            label="Handwriting Style",
                            choices=[
                                ("Style 107 (Neat Script)", 107),
                                ("Style 226 (Print)", 226),
                                ("Style 696 (Cursive)", 696),
                                ("Style 50 (Casual)", 50),
                                ("Style 300 (Slanted)", 300),
                                ("Random Style (-1)", -1),
                            ],
                            value=107,
                        )
                        beautify_input = gr.Checkbox(label="Beautify Strokes", value=True)
                    synth_btn = gr.Button("Generate Handwriting Preview", variant="primary")

                with gr.Column(scale=1):
                    svg_output = gr.HTML(label="Handwriting Preview")
                    info_output = gr.Markdown()

            synth_btn.click(
                fn=gradio_synthesize,
                inputs=[text_input, style_input, beautify_input],
                outputs=[svg_output, info_output],
            )

        app = gr.mount_gradio_app(app, demo, path="/")
    except ImportError:
        pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
