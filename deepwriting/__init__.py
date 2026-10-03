"""
DeepWriting: Digital Ink Generation & Handwriting Synthesis Library
"""
from deepwriting.writer import DeepWriter, load_model
from deepwriting.result import SynthesisResult

__version__ = "1.0.0"
__all__ = ["DeepWriter", "load_model", "SynthesisResult"]
