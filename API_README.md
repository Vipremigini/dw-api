# DeepWriting Headless REST API Reference

A high-performance, container-ready REST API for generating realistic online handwriting trajectories, single-line centerline vector graphics (SVG), high-resolution raster images (PNG), and raw pen coordinates from arbitrary text.

Designed for deployment on **Hugging Face Spaces (Free Tier CPU Basic • 16 GB RAM)** and local Docker/Kubernetes environments.

---

## Table of Contents

- [Quick Overview](#quick-overview)
- [Base URL & Interactive Documentation](#base-url--interactive-documentation)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
  - [1. Service Info & Discovery (`GET /`)](#1-service-info--discovery-get-)
  - [2. Health Check (`GET /health`)](#2-health-check-get-health)
  - [3. List Available Styles (`GET /v1/styles`)](#3-list-available-styles-get-v1styles)
  - [4. Synthesize Handwriting (`POST /v1/synthesize`)](#4-synthesize-handwriting-post-v1synthesize)
  - [5. Direct SVG Stream (`GET /v1/synthesize/svg`)](#5-direct-svg-stream-get-v1synthesizesvg)
  - [6. Direct PNG Stream (`GET /v1/synthesize/png`)](#6-direct-png-stream-get-v1synthesizepng)
- [Client Integration Examples](#client-integration-examples)
  - [cURL](#curl)
  - [JavaScript / TypeScript (Browser / Node.js)](#javascript--typescript-browser--nodejs)
  - [Python](#python)
- [Pen Plotter & Hardware CNC Integration](#pen-plotter--hardware-cnc-integration)
- [Local Running & Testing](#local-running--testing)
- [GitHub Codespaces Cloud Hosting (100% Free)](#github-codespaces-cloud-hosting-100-free--60-hoursmonth--no-credit-card)
- [Hugging Face Spaces Deployment Guide](#hugging-face-spaces-deployment-guide-100-free-tier)

---

## Quick Overview

Unlike conventional digital typography or raster vectorizers that produce closed 2D dual-line contours, DeepWriting natively generates **online single-stroke centerlines** using Conditional Variational Recurrent Neural Networks (C-VRNN).

- **Formats**: Single-line vector SVG, PNG images, and raw trajectory JSON.
- **Port**: `7860` (standard Hugging Face Spaces port).
- **Concurrency**: Thread-safe with inference queued via `asyncio.Lock()` and offloaded to non-blocking worker threads.
- **CORS**: Fully enabled (`allow_origins=["*"]`) for web and mobile frontends.

---

## Base URL & Interactive Documentation

| Environment | Base URL |
| :--- | :--- |
| **Local Dev** | `http://localhost:7860` |
| **Hugging Face Space** | `https://<USERNAME>-<SPACE_NAME>.hf.space` |

### Interactive API Explorers
- **Swagger UI (Interactive Playground)**: `/docs`
- **ReDoc (API Specification)**: `/redoc`
- **OpenAPI 3.1 JSON Schema**: `/openapi.json`

---

## Authentication

- **Public Hugging Face Space**: No authentication required.
- **Private Hugging Face Space**: Provide your Hugging Face User Access Token in the `Authorization` header:
  ```http
  Authorization: Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

---

## API Endpoints

### 1. Service Info & Discovery (`GET /`)

Returns API service metadata, ready state, and endpoint shortcuts.

#### Response (`200 OK`)
```json
{
  "service": "DeepWriting Headless REST API",
  "status": "ready",
  "documentation": "/docs",
  "openapi_schema": "/openapi.json",
  "endpoints": {
    "health": "/health",
    "styles": "/v1/styles",
    "synthesize_post": "POST /v1/synthesize",
    "synthesize_svg_get": "GET /v1/synthesize/svg?text=...&style=107",
    "synthesize_png_get": "GET /v1/synthesize/png?text=...&style=107"
  }
}
```

---

### 2. Health Check (`GET /health`)

Liveness and readiness probe for container orchestrators and monitoring tools.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
*Returns `503 Service Unavailable` if the model is still loading into memory during startup.*

---

### 3. List Available Styles (`GET /v1/styles`)

Returns the list of dataset style indices available for handwriting conditioning.

#### Response (`200 OK`)
```json
{
  "total_styles": 705,
  "styles": [0, 1, 2, "...", 704],
  "recommended": [107, 226, 696, 50, 300],
  "random_style_option": -1
}
```

---

### 4. Synthesize Handwriting (`POST /v1/synthesize`)

Main synthesis endpoint supporting multiple output formats: `svg`, `png`, and `json`.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body
```json
{
  "text": "DeepWriting generates single-line vector ink.",
  "style": 107,
  "format": "svg",
  "beautify": true,
  "stroke_width": 2.0,
  "color": "#000000",
  "background": null,
  "dpi": 200
}
```

#### Request Parameters Reference

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | `string` | **Required** | Text string to synthesize (1–500 chars). |
| `style` | `integer` | `107` | Style sample ID (`0` to `704`). Pass `null` or `-1` for random unbiased handwriting. |
| `format` | `string` | `"svg"` | Output format: `"svg"`, `"png"`, or `"json"`. |
| `beautify` | `boolean` | `true` | Applies mean-sampling filter for smoothed, neat handwriting strokes. |
| `stroke_width` | `float` | `2.0` | Stroke line width in points/pixels (`0.1` to `20.0`). |
| `color` | `string` | `"#000000"` | Ink color (hex code or CSS color name like `'navy'`). |
| `background` | `string` or `null` | `null` | Background color. Defaults to `null` (transparent for SVG, white for PNG). |
| `dpi` | `integer` | `200` | Resolution DPI for PNG format (`72` to `600`). |

#### Responses by Format

##### When `format="svg"`
- **Status**: `200 OK`
- **Content-Type**: `image/svg+xml`
- **Body**: Complete standalone SVG vector graphic XML string.

##### When `format="png"`
- **Status**: `200 OK`
- **Content-Type**: `image/png`
- **Body**: Binary PNG image byte stream.

##### When `format="json"`
- **Status**: `200 OK`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "text": "DeepWriting generates single-line vector ink.",
  "style_id": 107,
  "beautified": true,
  "stroke_count": 369,
  "strokes": [
    [0.012, -0.045, 0.0],
    [-0.008, 0.032, 0.0],
    [0.051, 0.002, 1.0]
  ],
  "lines": [
    [
      [12.4, 45.1],
      [14.2, 48.7],
      [16.8, 52.0]
    ]
  ],
  "svg": "<?xml version=\"1.0\" encoding=\"utf-8\" ?>..."
}
```

- **`strokes`**: Raw relative displacement array `[dx, dy, pen_up]` where `pen_up == 1.0` marks a pen lift.
- **`lines`**: Array of unbroken continuous coordinate segments `[[(x, y), ...], ...]`, ready for pen plotter toolpaths (G-code / HP-GL).

---

### 5. Direct SVG Stream (`GET /v1/synthesize/svg`)

Convenient endpoint for embedding handwriting directly into HTML, Markdown, or `<img>` elements without JavaScript.

#### Query Parameters
- `text` *(string, required)*: Text to generate.
- `style` *(int, default: 107)*: Style ID.
- `stroke_width` *(float, default: 2.0)*: Line width.
- `color` *(string, default: "#000000")*: Ink color.
- `background` *(string, optional)*: Background color (omit for transparent).
- `beautify` *(bool, default: true)*: Apply stroke smoothing.

#### HTML Usage Example
```html
<img src="https://<YOUR_SPACE>.hf.space/v1/synthesize/svg?text=Hello+World&style=107&color=darkblue" alt="Handwritten text" />
```

---

### 6. Direct PNG Stream (`GET /v1/synthesize/png`)

Directly renders and streams a high-resolution PNG image.

#### Query Parameters
- `text` *(string, required)*: Text to generate.
- `style` *(int, default: 107)*: Style ID.
- `linewidth` *(float, default: 2.0)*: Line width.
- `color` *(string, default: "#000000")*: Ink color.
- `dpi` *(int, default: 200)*: Output resolution DPI.
- `beautify` *(bool, default: true)*: Apply stroke smoothing.

---

## Client Integration Examples

### cURL

#### Download SVG Vector
```bash
curl -X POST "https://<YOUR_SPACE>.hf.space/v1/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello from DeepWriting", "style": 107, "format": "svg"}' \
     --output handwriting.svg
```

#### Download PNG Image via GET
```bash
curl -G "https://<YOUR_SPACE>.hf.space/v1/synthesize/png" \
     --data-urlencode "text=Realistic handwriting generated via API" \
     --data-urlencode "style=226" \
     --data-urlencode "color=#1a365d" \
     --output signature.png
```

#### Fetch Raw Coordinates (JSON)
```bash
curl -X POST "https://<YOUR_SPACE>.hf.space/v1/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Plotter coordinates", "format": "json"}'
```

---

### JavaScript / TypeScript (Browser / Node.js)

#### 1. Fetch and Display SVG in DOM
```javascript
async function generateHandwriting(text, style = 107) {
  const response = await fetch("https://<YOUR_SPACE>.hf.space/v1/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: text,
      style: style,
      format: "svg",
      stroke_width: 2.0,
      color: "#111827"
    })
  });

  const svgXml = await response.text();
  document.getElementById("handwriting-container").innerHTML = svgXml;
}
```

#### 2. Fetch PNG as Blob / Object URL
```javascript
async function getHandwritingImageUrl(text) {
  const response = await fetch(`https://<YOUR_SPACE>.hf.space/v1/synthesize/png?text=${encodeURIComponent(text)}&style=696`);
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}
```

---

### Python

```python
import requests

API_URL = "https://<YOUR_SPACE>.hf.space"

# 1. Check health
health = requests.get(f"{API_URL}/health").json()
print("API status:", health)

# 2. Synthesize to SVG
payload = {
    "text": "Plug and play headless API integration.",
    "style": 107,
    "format": "svg",
    "stroke_width": 2.2,
    "color": "#0d47a1"
}

res = requests.post(f"{API_URL}/v1/synthesize", json=payload)
with open("result.svg", "w", encoding="utf-8") as f:
    f.write(res.text)
print("Saved result.svg")

# 3. Fetch trajectory points for custom applications
json_res = requests.post(f"{API_URL}/v1/synthesize", json={"text": "Coordinates", "format": "json"}).json()
print(f"Generated {json_res['stroke_count']} stroke points in {len(json_res['lines'])} lines")
```

---

## Pen Plotter & Hardware CNC Integration

Standard fonts outline characters with two parallel closed paths. Pen plotters (AxiDraw, Cricut, LaserGRBL, CNC) require **single-line centerlines**.

The DeepWriting API outputs true single-line centerlines natively:

```python
import requests

response = requests.post("https://<YOUR_SPACE>.hf.space/v1/synthesize", json={
    "text": "Single line plotter test",
    "style": 107,
    "format": "json"
}).json()

# Convert lines into G-code
gcode = ["G21 ; Millimeter units", "G90 ; Absolute positioning", "G0 Z5.0 ; Pen Up"]
for stroke in response["lines"]:
    # Move pen to start of unbroken stroke
    x0, y0 = stroke[0]
    gcode.append(f"G0 X{x0:.2f} Y{y0:.2f}")
    gcode.append("G1 Z0.0 F100.0 ; Pen Down")
    
    # Draw along trajectory
    for x, y in stroke[1:]:
        gcode.append(f"G1 X{x:.2f} Y{y:.2f} F500.0")
        
    gcode.append("G0 Z5.0 ; Pen Up")

with open("output.gcode", "w") as f:
    f.write("\n".join(gcode))
```

---

## Local Running & Testing

### Running with Python
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Uvicorn
python app.py
# Or:
uvicorn app:app --host 0.0.0.0 --port 7860
```
Open [http://localhost:7860/docs](http://localhost:7860/docs) in your browser.

### Running with Docker Locally
```bash
# 1. Build Docker image
docker build -t deepwriting-api .

# 2. Run container
docker run -p 7860:7860 deepwriting-api
```

---

## GitHub Codespaces Cloud Hosting (100% Free • 60 Hours/Month • No Credit Card)

Every GitHub account includes **60 free hours of cloud compute each month** with **8 GB RAM**. You can host and run this headless API directly in GitHub's cloud and expose it with a public HTTPS URL.

### 1. Launch a Codespace
1. Push this repository to your GitHub account (or navigate to your repository on github.com).
2. Click the green **Code** button at the top right.
3. Switch to the **Codespaces** tab and click **Create codespace on main**.
4. GitHub will boot a cloud Linux environment and automatically install all dependencies via `.devcontainer/devcontainer.json`.

### 2. Start the API
In the built-in Codespace terminal:
```bash
python app.py
```
*(The API starts on port `7860`)*

### 3. Make the Port Public
Because this repo includes `.devcontainer/devcontainer.json`, Port `7860` is automatically forwarded. To verify or ensure it is public:
1. In the bottom panel of VS Code in your browser, switch to the **Ports** tab.
2. Locate Port **`7860`**.
3. Right-click on Port `7860` ➔ set **Port Visibility** to **Public**.
4. Copy the forwarded address. It will look like:
   ```
   https://<your-codespace-name>-7860.app.github.dev
   ```

### 4. Call Your Cloud API
Your API is now accessible globally from any frontend, curl, or mobile app:
- **Interactive Swagger Docs**: `https://<your-codespace-name>-7860.app.github.dev/docs`
- **Health Check**: `https://<your-codespace-name>-7860.app.github.dev/health`
- **SVG Generation**:
  ```bash
  curl -X POST "https://<your-codespace-name>-7860.app.github.dev/v1/synthesize" \
       -H "Content-Type: application/json" \
       -d '{"text": "Hosted on GitHub Codespaces", "style": 107, "format": "svg"}' \
       --output handwriting.svg
  ```

---

## Hugging Face Spaces Deployment Guide (100% Free Tier)

Hugging Face Spaces offers a completely free CPU tier with **2 vCPU, 16 GB RAM, and 50 GB disk** using the **Gradio SDK** (requires zero payment or credit card).

Because our application mounts FastAPI directly inside the Gradio runtime, you get both the full **Headless REST API** (`/v1/synthesize`, `/health`, `/docs`) and an interactive browser testing playground on the 100% free tier.

### Step-by-Step Deployment
1. **Create Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Enter Space name (e.g. `deepwriting-api`).
   - Select **Gradio** as the Space SDK (100% Free, no credit card required).
   - Select **CPU Basic • 16 GB RAM • Free** hardware.

2. **Ensure Git LFS is Active**:
   The model weights (`model-108480.data-00000-of-00001`) are ~83 MB and tracked via `.gitattributes`.
   ```bash
   git lfs install
   ```

3. **Push Repository to Hugging Face**:
   ```bash
   git branch -M main
   git add .
   git commit -m "Deploy DeepWriting headless REST API to Hugging Face Free Tier"
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
   git push space main
   ```

4. **Monitor Startup & Access**:
   Hugging Face installs dependencies from `requirements.txt`, boots `app.py`, and exposes your service at:
   - **Base URL**: `https://<YOUR_USERNAME>-<YOUR_SPACE_NAME>.hf.space`
   - **Interactive API Docs**: `https://<YOUR_USERNAME>-<YOUR_SPACE_NAME>.hf.space/docs`
   - **Health Check**: `https://<YOUR_USERNAME>-<YOUR_SPACE_NAME>.hf.space/health`
