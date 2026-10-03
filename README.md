---
title: DeepWriting Headless API
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# DeepWriting: Handwriting Synthesis & Editing

Code and pretrained models for **[DeepWriting: Making Digital Ink Editable via Deep Generative Modeling](https://arxiv.org/abs/1801.08379)** (CHI '18).

Implementation of Conditional Variational Recurrent Neural Networks (C-VRNN) with Gaussian Mixture Model (GMM) latent spaces for handwriting synthesis, generation, beautification, and editing.

> **Modernized Edition:** Updated with Apple Silicon (M-series / ARM64) support, TensorFlow 2.x compatibility bridge, modern SciPy/Matplotlib/Scikit-Learn integration, and a standalone handwriting synthesis CLI (`synthesize.py`).

---

## Quick Start (Handwriting Generation)

### 1. Installation

Set up a virtual environment with Python 3.10+ (tested on Python 3.11):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Python Library Usage (Plug & Play)

> 📖 **Complete Documentation:** See **[`LIBRARY.md`](LIBRARY.md)** for full API reference, methods, web framework examples (FastAPI/Flask), and plotter/CNC integration.

Install in editable mode:
```bash
pip install -e .
```

### Option A: Context Manager (Auto-Unload)
The model is loaded into memory, synthesizes text, and automatically unloads when exiting the block:

```python
import deepwriting

with deepwriting.load_model() as writer:
    result = writer.synthesize("DeepWriting plug and play", style=107)
    
    # Save vector SVG or high-res PNG
    result.save("sample.png")
    result.save("sample.svg")
    
    # Or get in-memory PIL Image (no disk writing)
    pil_image = result.to_image()

# Model session and graph are automatically freed from RAM here!
```

### Option B: Explicit Lifecycle (Load -> Generate Fast -> Unload)
Load the model once, generate multiple sentences rapidly with zero reload overhead, then manually unload:

```python
import deepwriting

# 1. Load model into memory
writer = deepwriting.load_model()

# 2. Fast continuous generation across different styles
res1 = writer.synthesize("First sentence in neat style.", style=107)
res1.save("sentence1.png")

res2 = writer.synthesize("Second sentence in cursive style.", style=696)
res2.save("sentence2.png")

# Inspect raw stroke offsets (shape: N x 3 -> [dx, dy, pen_up])
print(res2.strokes)

# 3. Unload model and free RAM
writer.unload()
```

### Single-Line Vector Export (Pen Plotters & Laser Cutters)

Unlike standard digital fonts that use closed dual-line outline contours, DeepWriting natively predicts online pen movement trajectories (centerline paths).

You can export true single-line SVGs (`fill="none"`, `stroke-linecap="round"`) directly for **AxiDraw, Cricut, LaserGRBL, LightBurn, CNC**, or CAD:

```python
# Save transparent single-line vector SVG for pen plotters / laser cutters
res.save_svg("plotter_output.svg", stroke_width=2.0)

# Or export with custom ink color and background
res.save_svg("whiteboard.svg", stroke_width=2.5, color="#0052cc", background="white")

# Access raw continuous pen lines for G-code / HP-GL generation:
for line_idx, line in enumerate(res.lines):
    # Each line is a numpy array of (x, y) coordinates for an unbroken stroke
    print(f"Stroke {line_idx}: {len(line)} points")
```

---

## Command-Line Usage (CLI)

Use the included `synthesize.py` tool to generate realistic handwriting directly from the terminal:

```bash
# Synthesize text with a specific style (e.g., style ID 107, 226, 696)
python synthesize.py -t "Hello world from DeepWriting" -s 107 -o output/my_handwriting

# Synthesize with random unbiased style
python synthesize.py -t "DeepWriting online handwriting synthesis" -s -1 -o output/random_style

# Disable handwriting beautification filter
python synthesize.py -t "Raw personal handwriting" -s 226 --no-beautify -o output/raw_style
```

---

## Dataset & Pretrained Model

- **Pretrained Synthesis Model:** [Download Model](https://files.ait.ethz.ch/projects/deepwriting/tf-1514981744-deepwriting_synthesis_model.tar.gz) (uncompress into `pretrained_models/`)
- **Dataset:** [Download Preprocessed Data](https://files.ait.ethz.ch/projects/deepwriting/deepwriting_dataset.zip) (unzip into `data/`)

The dataset compiles handwriting from 294 authors based on the [IAMOnDB](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) corpus, segmented into ~300 stroke samples with offset representations.

---

## Model Evaluation

To reproduce paper evaluation benchmarks or run bulk qualitative evaluation:

```bash
export PYTHONPATH=$PYTHONPATH:./source
python tf_evaluate_hw.py -S pretrained_models/ -M tf-1514981744-deepwriting_synthesis_model -QL
```

Evaluation options and sampling hyper-parameters can be adjusted globally in `tf_evaluate_hw.py`.

---

## Model Training

1. Training details and hyper-parameters are configured in `config.py`.
2. Ensure `data/deepwriting_training.npz` and `data/deepwriting_validation.npz` are present in `data/`.
3. Set `PYTHONPATH`:
   ```bash
   export PYTHONPATH=$PYTHONPATH:./source
   ```
4. Start training:
   ```bash
   python tf_train_hw.py -S ./runs
   ```
5. Resume training from checkpoint:
   ```bash
   python tf_train_hw.py -S ./runs -M <model_folder_name>
   ```

---

## Headless REST API (Hugging Face Spaces)

This repository includes a production-ready headless API (`app.py`) built with FastAPI, designed for **Hugging Face Spaces (Docker / Free Tier - 2 vCPU, 16 GB RAM)**.

> 📖 **Complete API Reference:** See **[`API_README.md`](API_README.md)** for exhaustive endpoint specifications, request/response JSON schemas, cURL commands, client SDK integration (JS/Python), and pen plotter / CNC G-code examples.

### Running Locally
```bash
python app.py
# Or with uvicorn:
uvicorn app:app --host 0.0.0.0 --port 7860
```
Interactive OpenAPI documentation will be accessible at `http://localhost:7860/docs`.

### API Endpoints
- `GET /`: Health status, available endpoints, and metadata.
- `GET /health`: Health probe (`{"status": "healthy", "model_loaded": true}`).
- `GET /v1/styles`: List of 700+ available handwriting style indices.
- `POST /v1/synthesize`: Synthesize text to `svg`, `png`, or structured `json` stroke coordinates.
- `GET /v1/synthesize/svg?text=...&style=107`: Direct single-line SVG stream (embeddable in `<img>` tags).
- `GET /v1/synthesize/png?text=...&style=107`: Direct PNG image stream.

### Cloud Hosting: GitHub Codespaces (100% Free • No Credit Card)

1. Open this repository on **GitHub**.
2. Click **Code** ➔ **Codespaces** ➔ **Create codespace on main** (every GitHub account gets 60 hours/month free with 8 GB RAM).
3. The environment will automatically build and install all dependencies via `.devcontainer/devcontainer.json`.
4. In the Codespace terminal, run:
   ```bash
   python app.py
   ```
5. In the **Ports** tab, Port `7860` will be forwarded publicly as `https://<codespace-name>-7860.app.github.dev`.

---

## Key Files Added / Modernized

- `compat.py`: Transparent compatibility layer bridging TensorFlow 1.x graph execution, `tf.contrib.rnn`, `tf.contrib.distributions`, and SciPy image routines to modern environments.
- `synthesize.py`: Standalone CLI tool for fast sentence-level handwriting synthesis to SVG and PNG.
- `requirements.txt`: Curated dependency specification.
- `visualize_hw.py`: Fixed SVG stroke path move command syntax.
- `dataset_hw.py`: Updated `OneHotEncoder` parameters for modern Scikit-Learn versions.

---

## Citation & Attribution

If you use this code or dataset in your research, please cite the original authors:

```bibtex
@inproceedings{Aksan:2018:DeepWriting,
  author    = {Aksan, Emre and Pece, Fabrizio and Hilliges, Otmar},
  title     = {{DeepWriting: Making Digital Ink Editable via Deep Generative Modeling}},
  booktitle = {Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems},
  series    = {CHI '18},
  year      = {2018},
  location  = {Montreal QC, Canada},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```

The codebase is released under the [MIT License](LICENSE).
