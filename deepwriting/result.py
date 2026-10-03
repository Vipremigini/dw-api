"""
SynthesisResult encapsulates the output of the DeepWriting synthesis model.
Provides methods to inspect strokes and export to SVG, PNG, or in-memory PIL Image.
"""
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import svgwrite


class SynthesisResult:
    """Encapsulates synthesized handwriting stroke data and rendering methods."""

    def __init__(self, strokes, text, style_id=None, beautified=True):
        """
        Args:
            strokes (np.ndarray): Array of shape (N, 3) where columns are [dx, dy, pen_up].
            text (str): The text string synthesized.
            style_id (int, optional): Reference style sample index used.
            beautified (bool): Whether beautification was applied.
        """
        self._strokes = np.array(strokes, dtype=np.float32)
        self.text = text
        self.style_id = style_id
        self.beautified = beautified

    @property
    def strokes(self):
        """Returns raw numpy array of stroke offsets with shape (N, 3): [dx, dy, pen_up]."""
        return self._strokes

    def __len__(self):
        return len(self._strokes)

    def __repr__(self):
        return (f"<SynthesisResult text='{self.text}' style={self.style_id} "
                f"strokes={len(self._strokes)} beautified={self.beautified}>")

    @property
    def lines(self):
        """
        Returns strokes grouped into continuous line segments as a list of (M, 2) numpy arrays.
        Each element represents an unbroken pen stroke from pen-down to pen-up.
        Useful for custom G-code, HP-GL (pen plotters), CNC, or shapely / geometry pipelines.
        """
        return self._get_lines()

    def get_absolute_strokes(self, factor=0.001):
        """
        Returns an array of shape (N, 3) containing [abs_x, abs_y, pen_up].
        """
        abs_x = 0.0
        abs_y = 0.0
        result = []
        for idx in range(len(self._strokes)):
            abs_x += float(self._strokes[idx, 0]) / factor
            abs_y += float(self._strokes[idx, 1]) / factor
            result.append([abs_x, -abs_y, self._strokes[idx, 2]])
        return np.array(result, dtype=np.float32)

    def _get_lines(self, factor=0.001):
        """Helper to convert relative stroke offsets into list of continuous stroke lines."""
        abs_x = 0.0
        abs_y = 0.0
        lines = []
        current_line = []

        for idx in range(len(self._strokes)):
            dx = float(self._strokes[idx, 0]) / factor
            dy = float(self._strokes[idx, 1]) / factor
            abs_x += dx
            abs_y += dy
            current_line.append((abs_x, -abs_y))  # Invert Y so handwriting is right-side up

            if self._strokes[idx, 2] == 1:  # Pen up
                if len(current_line) > 1:
                    lines.append(np.array(current_line))
                current_line = []

        if len(current_line) > 1:
            lines.append(np.array(current_line))

        return lines

    def to_image(self, linewidth=2.0, color='black', factor=0.001, dpi=200):
        """
        Renders the handwriting into an in-memory PIL Image object.
        Ideal for web applications, APIs, or immediate display without touching the disk.

        Returns:
            PIL.Image.Image: Rendered image.
        """
        buf = io.BytesIO()
        self.save_png(buf, linewidth=linewidth, color=color, factor=factor, dpi=dpi)
        buf.seek(0)
        return Image.open(buf)

    def _repr_png_(self):
        """Enables rich visual rendering automatically in Jupyter Notebooks."""
        buf = io.BytesIO()
        self.save_png(buf, linewidth=2.0, color='black', factor=0.001, dpi=150)
        return buf.getvalue()

    def save_png(self, output_target, linewidth=2.0, color='black', factor=0.001, dpi=200):
        """
        Saves or writes the handwriting as a high-resolution PNG image.

        Args:
            output_target (str or file-like): Filepath or buffer to save PNG to.
            linewidth (float): Stroke line width.
            color (str): Ink color (e.g. 'black', '#1a365d', 'midnightblue').
            factor (float): Scale factor for coordinate normalization.
            dpi (int): Image resolution.
        """
        lines = self._get_lines(factor=factor)
        if not lines:
            raise ValueError("No drawable strokes found in synthesis result.")

        all_pts = np.concatenate(lines, axis=0)
        width = all_pts[:, 0].max() - all_pts[:, 0].min()
        height = all_pts[:, 1].max() - all_pts[:, 1].min()

        aspect_ratio = max(width / max(height, 1.0), 1.0)
        fig_w = min(max(6.0, aspect_ratio * 1.8), 24.0)
        fig_h = max(2.5, fig_w / aspect_ratio)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=linewidth, solid_capstyle='round')

        ax.axis('off')
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()

        if isinstance(output_target, str):
            parent_dir = os.path.dirname(output_target)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            plt.savefig(output_target, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        else:
            plt.savefig(output_target, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='white')

        plt.close(fig)

    def save_svg(self, output_path, factor=0.001, stroke_width=2.0, color='black', background=None):
        """
        Saves the handwriting as a true single-line vector graphic (SVG).

        Unlike dual-line font outlines, this outputs true open centerline stroke paths
        (`fill="none"` and `stroke-linecap="round"`), making it directly compatible
        with pen plotters (e.g. AxiDraw), laser cutters/engravers, CNC, and CAD software.

        Args:
            output_path (str): Filepath to save SVG to.
            factor (float): Scale factor for coordinate normalization.
            stroke_width (float): Vector stroke line width in SVG units.
            color (str): Stroke color (e.g. 'black', '#000000', 'blue').
            background (str, optional): Background color (e.g. 'white').
                Defaults to None (transparent background, ideal for plotters).
        """
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        lines = self._get_lines(factor=factor)
        if not lines:
            raise ValueError("No drawable strokes found in synthesis result.")

        all_pts = np.concatenate(lines, axis=0)
        min_x = float(all_pts[:, 0].min())
        max_x = float(all_pts[:, 0].max())
        min_y = float(all_pts[:, 1].min())
        max_y = float(all_pts[:, 1].max())

        padding = 20.0
        width = max(max_x - min_x + 2 * padding, 10.0)
        height = max(max_y - min_y + 2 * padding, 10.0)

        dwg = svgwrite.Drawing(output_path, size=(f"{width:.1f}px", f"{height:.1f}px"),
                              viewBox=f"0 0 {width:.1f} {height:.1f}")

        if background:
            dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=background))

        # Build clean, connected single-line centerline vector paths
        for line in lines:
            # Shift coordinates by padding and min bounds
            # Invert back Y to SVG coordinate system (where Y grows downwards)
            pts = [(x - min_x + padding, -(y) - (-max_y) + padding) for x, y in line]
            path_data = [f"M {pts[0][0]:.2f},{pts[0][1]:.2f}"]
            for pt in pts[1:]:
                path_data.append(f"L {pt[0]:.2f},{pt[1]:.2f}")

            d = " ".join(path_data)
            path = dwg.path(
                d=d,
                fill="none",
                stroke=color,
                stroke_width=stroke_width,
                stroke_linecap="round",
                stroke_linejoin="round"
            )
            dwg.add(path)

        dwg.save()

    def to_svg_string(self, factor=0.001, stroke_width=2.0, color='black', background=None):
        """
        Returns the single-line vector handwriting as an in-memory SVG string.
        """
        import io
        buf = io.StringIO()
        # svgwrite Drawing can write to any stream
        dwg = svgwrite.Drawing()
        lines = self._get_lines(factor=factor)
        if not lines:
            return ""

        all_pts = np.concatenate(lines, axis=0)
        min_x = float(all_pts[:, 0].min())
        max_x = float(all_pts[:, 0].max())
        min_y = float(all_pts[:, 1].min())
        max_y = float(all_pts[:, 1].max())

        padding = 20.0
        width = max(max_x - min_x + 2 * padding, 10.0)
        height = max(max_y - min_y + 2 * padding, 10.0)

        dwg['width'] = f"{width:.1f}px"
        dwg['height'] = f"{height:.1f}px"
        dwg['viewBox'] = f"0 0 {width:.1f} {height:.1f}"

        if background:
            dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=background))

        for line in lines:
            pts = [(x - min_x + padding, -(y) - (-max_y) + padding) for x, y in line]
            path_data = [f"M {pts[0][0]:.2f},{pts[0][1]:.2f}"]
            for pt in pts[1:]:
                path_data.append(f"L {pt[0]:.2f},{pt[1]:.2f}")

            d = " ".join(path_data)
            path = dwg.path(
                d=d,
                fill="none",
                stroke=color,
                stroke_width=stroke_width,
                stroke_linecap="round",
                stroke_linejoin="round"
            )
            dwg.add(path)

        dwg.write(buf)
        return buf.getvalue()

    def save(self, filepath, **kwargs):
        """
        Convenience method to save output based on file extension (.svg or .png).
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.svg':
            self.save_svg(filepath, **kwargs)
        elif ext in ('.png', '.jpg', '.jpeg'):
            self.save_png(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Use .svg or .png")
