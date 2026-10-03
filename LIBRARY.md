# DeepWriting Python Library Documentation

A clean, plug-and-play Python library for conditional handwriting generation from text using deep recurrent neural networks.

Natively generates **online pen stroke trajectories**, enabling export to high-resolution raster images (PNG) and **true single-line centerline vector graphics (SVG)** for pen plotters, laser engravers, CNC, and CAD.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Pattern 1: Context Manager (Auto-Unload)](#pattern-1-context-manager-auto-unload)
  - [Pattern 2: Persistent Server / API Lifecycle](#pattern-2-persistent-server--api-lifecycle)
- [API Reference](#api-reference)
  - [deepwriting.load_model()](#deepwritingload_model)
  - [DeepWriter Class](#deepwriter-class)
  - [SynthesisResult Class](#synthesisresult-class)
- [Single-Line Vectors & Plotters (AxiDraw / CNC)](#single-line-vectors--plotters-axidraw--cnc)
- [Web API Integration (FastAPI / Flask)](#web-api-integration-fastapi--flask)
- [Alphabet & Character Set](#alphabet--character-set)
- [Memory Management](#memory-management)

---

## Installation

Install the package into your Python environment from the repository root:

```bash
# Editable install (recommended for local development)
pip install -e .

# Or install directly from another directory
pip install /path/to/deepwriting
```

### Requirements
- Python 3.8+ (tested on Python 3.11 ARM64 on macOS / Apple Silicon and Linux x86_64)
- TensorFlow >= 2.0 (handled automatically via built-in compatibility layer)
- NumPy, Matplotlib, Pillow, svgwrite, scikit-learn

---

## Quick Start

### Pattern 1: Context Manager (Auto-Unload)

Recommended for standalone scripts, batch processing, or Jupyter Notebooks. The model loads into RAM and is automatically unloaded when the `with` block exits:

```python
import deepwriting

with deepwriting.load_model() as writer:
    # Generate handwriting
    res = writer.synthesize("Hello from DeepWriting!", style=107)

    # Save outputs
    res.save("output.png")
    res.save("output.svg")

# TensorFlow session and memory are automatically released here!
```

---

### Pattern 2: Persistent Server / API Lifecycle

Recommended for web services (FastAPI, Flask), background workers, or interactive applications. Load the model once at startup and run rapid continuous generation without reloading overhead:

```python
import deepwriting

# 1. Load model once at startup (~2-3 seconds)
writer = deepwriting.load_model()

# 2. Fast subsequent synthesis with zero reload latency
res1 = writer.synthesize("First sentence in print style.", style=107)
res1.save("sentence1.png")

res2 = writer.synthesize("Second sentence in cursive style.", style=696)
res2.save("sentence2.svg")

# 3. Unload when stopping the application / server shutdown
writer.unload()
```

---

## API Reference

### `deepwriting.load_model`

```python
deepwriting.load_model(model_dir=None, data_file=None) -> DeepWriter
```

Loads the pretrained DeepWriting synthesis model into memory and returns a `DeepWriter` instance.

#### Parameters:
- `model_dir` *(str, optional)*: Path to directory containing model checkpoint. Defaults to `pretrained_models/tf-1514981744-deepwriting_synthesis_model/`.
- `data_file` *(str, optional)*: Path to training `.npz` dataset (used for normalization parameters and reference styles). Defaults to `data/deepwriting_training.npz`.

#### Returns:
- `DeepWriter`: An active synthesizer instance.

---

### `DeepWriter` Class

Encapsulates the TensorFlow computational graph and model weights.

#### Methods:

##### `writer.synthesize(text, style=None, beautify=True, timeout_sec=30.0) -> SynthesisResult`
Synthesizes handwritten pen strokes for the given input text.

- **`text`** *(str)*: Text string to synthesize (alphanumeric and standard punctuation).
- **`style`** *(int, optional)*: Style sample ID from the dataset (e.g., `107`, `226`, `696`). If `-1` or `None`, a random unbiased style is used.
- **`beautify`** *(bool, default=True)*: Applies post-synthesis stroke smoothing and alignment normalization.
- **`timeout_sec`** *(float, default=30.0)*: Safety timeout in seconds to prevent runaway generation loops.

##### `writer.unload()`
Closes the active TensorFlow session, clears the default computation graph, and invokes Python garbage collection to free RAM.

---

### `SynthesisResult` Class

Returned by `writer.synthesize()`. Encapsulates stroke coordinates and provides export utilities.

#### Attributes & Properties:

| Attribute / Property | Type | Description |
| :--- | :--- | :--- |
| `res.text` | `str` | The input text that was synthesized. |
| `res.style_id` | `int` or `None` | The reference style sample index used. |
| `res.beautified` | `bool` | Whether stroke beautification was applied. |
| `res.strokes` | `np.ndarray` | Raw shape `(N, 3)` array: `[dx, dy, pen_up]` relative coordinate offsets. |
| `res.lines` | `List[np.ndarray]` | Strokes grouped into continuous line segments `[(M, 2), ...]`. Each segment represents an unbroken pen path from pen-down to pen-up. |

#### Methods:

##### `res.save(filepath, **kwargs)`
Convenience method that saves either PNG or SVG based on the file extension:
```python
res.save("document.png")  # Calls save_png
res.save("document.svg")  # Calls save_svg
```

##### `res.save_svg(output_path, stroke_width=2.0, color='black', background=None, factor=0.001)`
Exports a **true single-line centerline vector graphic (SVG)** (`fill="none"`, `stroke-linecap="round"`).
- **`output_path`** *(str)*: Target SVG filepath.
- **`stroke_width`** *(float, default=2.0)*: SVG path stroke width.
- **`color`** *(str, default='black')*: Ink color (e.g. `'#000000'`, `'navy'`).
- **`background`** *(str, optional)*: Background fill color (e.g. `'white'`). Defaults to `None` (transparent, ideal for plotters).

##### `res.save_png(output_target, linewidth=2.0, color='black', factor=0.001, dpi=200)`
Renders and saves a high-resolution PNG image.
- **`output_target`** *(str or file-like buffer)*: Filepath or `io.BytesIO` buffer.
- **`linewidth`** *(float, default=2.0)*: Rendered stroke width.
- **`color`** *(str, default='black')*: Ink color.
- **`dpi`** *(int, default=200)*: Resolution.

##### `res.to_image(linewidth=2.0, color='black', factor=0.001, dpi=200) -> PIL.Image.Image`
Renders the handwriting directly into an in-memory **Pillow (`PIL.Image`)** object without touching disk.

##### `res.to_svg_string(stroke_width=2.0, color='black', background=None) -> str`
Returns the raw SVG XML string directly in memory.

##### `res.get_absolute_strokes(factor=0.001) -> np.ndarray`
Returns an array of shape `(N, 3)` containing absolute coordinates `[abs_x, abs_y, pen_up]`.

---

## Single-Line Vectors & Plotters (AxiDraw / CNC)

Unlike standard TTF/OTF fonts or image-tracing tools (which output closed, dual-line outline contours), DeepWriting natively outputs **centerline stroke paths**.

### Generating SVGs for Plotters
```python
res.save_svg("plotter_ready.svg", stroke_width=1.5)
```
Output SVG characteristics:
- `fill="none"`: No black polygons or interior fills.
- `stroke-linecap="round"` & `stroke-linejoin="round"`: Smooth pen lifts and corners.
- Continuous path commands (`M x y L x y ...`): No fragmented micro-segments; prevents plotter stutter.

### Direct G-code / Machine Trajectories
You can iterate directly over `res.lines` to drive hardware or custom kinematics:

```python
for stroke_idx, stroke_pts in enumerate(res.lines):
    # Move pen up to start of stroke
    x0, y0 = stroke_pts[0]
    print(f"G0 Z5.0")          # Pen Up
    print(f"G0 X{x0:.2f} Y{y0:.2f}")  # Travel
    print(f"G1 Z0.0 F100.0")   # Pen Down

    # Draw along stroke
    for x, y in stroke_pts[1:]:
        print(f"G1 X{x:.2f} Y{y:.2f} F500.0")

print(f"G0 Z5.0")  # Final Pen Up
```

---

## Web API Integration (FastAPI / Flask)

Because `deepwriting` supports in-memory image generation and persistent model sessions, it integrates cleanly into web services.

### FastAPI Example

```python
from fastapi import FastAPI, Response
import deepwriting

app = FastAPI()
writer = None

@app.on_event("startup")
def startup_event():
    global writer
    writer = deepwriting.load_model()

@app.on_event("shutdown")
def shutdown_event():
    global writer
    if writer:
        writer.unload()

@app.get("/generate/png")
def generate_png(text: str, style: int = 107):
    res = writer.synthesize(text, style=style)
    # Stream in-memory PNG without writing to disk
    import io
    buf = io.BytesIO()
    res.save_png(buf)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/generate/svg")
def generate_svg(text: str, style: int = 107):
    res = writer.synthesize(text, style=style)
    svg_str = res.to_svg_string()
    return Response(content=svg_str, media_type="image/svg+xml")
```

---

## Alphabet & Character Set

The pretrained model was trained on the IAM-OnDB dataset consisting of 70 distinct characters:

```text
0123456789
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
' . , - ( ) /
```

The library automatically sanitizes incoming text (e.g. converting `!` or `?` to `.`, `"` to `'`) to guarantee synthesis without `KeyError` exceptions.

---

## Memory Management

| Action | Memory Impact |
| :--- | :--- |
| `deepwriting.load_model()` | Allocates ~83 MB of model weights and TensorFlow runtime structures. |
| `writer.synthesize()` | Minimal temporary allocation during inference (~5-10 MB), freed after each run. |
| `writer.unload()` | Explicitly closes session, resets default graph, and calls `gc.collect()`. |
| `with deepwriting.load_model() as writer:` | Guarantees cleanup upon exiting the block, even if an exception occurs. |
