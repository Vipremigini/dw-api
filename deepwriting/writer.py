"""
DeepWriter handles the lifecycle of the DeepWriting model.
Supports loading the pretrained C-VRNN checkpoint, synthesizing sentences,
and completely unloading the session & graph from memory.
"""
import gc
import json
import os
import sys

# Ensure repository root and source directories are in sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_source_dir = os.path.join(_repo_root, 'source')
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _source_dir not in sys.path:
    sys.path.insert(0, _source_dir)

# Initialize compatibility bridge
import compat
import tensorflow as tf
import numpy as np

from tf_dataset_hw import HandWritingDatasetConditionalTF, HandWritingDatasetConditional, HandWritingDataset
from tf_models import VRNNGMM
from tf_models_hw import HandwritingVRNNGmmModel, HandwritingVRNNModel
from deepwriting.result import SynthesisResult


def _find_default_path(subpaths):
    """Searches candidates across repo root and current working directory."""
    cwd = os.getcwd()
    candidates = []
    for sp in subpaths:
        candidates.append(os.path.join(_repo_root, sp))
        candidates.append(os.path.join(cwd, sp))
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


class DeepWriter:
    """
    Plug-and-play handwriting synthesizer.
    Loads the C-VRNN generative model into memory once and allows fast continuous synthesis.
    """

    def __init__(self, model_dir=None, data_file=None, verbose=False):
        """
        Initializes and loads the pretrained DeepWriting model.

        Args:
            model_dir (str, optional): Path to the pretrained model folder containing config.json and checkpoint.
            data_file (str, optional): Path to deepwriting_validation.npz dataset statistics.
            verbose (bool): Whether to log detailed TensorFlow messages.
        """
        if not verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if model_dir is None:
            model_dir = _find_default_path([
                'pretrained_models/tf-1514981744-deepwriting_synthesis_model',
                'models/tf-1514981744-deepwriting_synthesis_model'
            ])
            if model_dir is None:
                raise FileNotFoundError(
                    "Default pretrained model not found. Please specify model_dir explicitly."
                )

        if data_file is None:
            data_file = _find_default_path([
                'data/deepwriting_validation.npz',
                'deepwriting/data/deepwriting_validation.npz'
            ])
            if data_file is None:
                raise FileNotFoundError(
                    "Validation dataset file not found. Please specify data_file explicitly."
                )

        self.model_dir = os.path.abspath(model_dir)
        self.data_file = os.path.abspath(data_file)
        self.verbose = verbose

        self.is_loaded = False
        self.sess = None
        self.graph = None
        self.dataset = None
        self.inference_model = None
        self.sampling_model = None

        self._load()

    def _load(self):
        """Builds computation graph and restores checkpoint weights."""
        if self.is_loaded:
            return

        config_path = os.path.join(self.model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {self.model_dir}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.config['model_dir'] = self.model_dir
        self.config['checkpoint_id'] = None

        # Create an isolated TensorFlow Graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            Dataset_cls = getattr(sys.modules[__name__], self.config['dataset_cls'])
            Model_cls = getattr(sys.modules[__name__], self.config['model_cls'])

            if issubclass(Dataset_cls, HandWritingDatasetConditional):
                self.dataset = Dataset_cls(self.data_file, var_len_seq=True, use_bow_labels=self.config['use_bow_labels'])
            elif issubclass(Dataset_cls, HandWritingDataset):
                self.dataset = Dataset_cls(self.data_file, var_len_seq=True)
            else:
                raise ValueError("Unknown dataset class.")

            batch_size = 1
            data_sequence_length = None

            self._ph_strokes = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(self.dataset.input_dims)])
            self._ph_targets = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(self.dataset.target_dims)])
            self._ph_seq_len = tf.placeholder(tf.int32, shape=[batch_size])

            # Build inference graph (used for style conditioning)
            with tf.name_scope("validation"):
                self.inference_model = Model_cls(
                    self.config,
                    reuse=False,
                    input_op=self._ph_strokes,
                    target_op=self._ph_targets,
                    input_seq_length_op=self._ph_seq_len,
                    input_dims=self.dataset.input_dims,
                    target_dims=self.dataset.target_dims,
                    batch_size=batch_size,
                    mode="validation",
                    data_processor=self.dataset
                )
                self.inference_model.build_graph()

            # Build sampling graph
            with tf.name_scope("sampling"):
                self.sampling_model = Model_cls(
                    self.config,
                    reuse=True,
                    input_op=self._ph_strokes,
                    target_op=None,
                    input_seq_length_op=self._ph_seq_len,
                    input_dims=self.dataset.input_dims,
                    target_dims=self.dataset.target_dims,
                    batch_size=batch_size,
                    mode="sampling",
                    data_processor=self.dataset
                )
                self.sampling_model.build_graph()

            # Create Session and restore parameters
            self.sess = tf.Session(graph=self.graph)
            saver = tf.train.Saver()
            checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {self.model_dir}")

            saver.restore(self.sess, checkpoint_path)

        self.is_loaded = True
        if self.verbose:
            print(f"[DeepWriter] Loaded checkpoint: {checkpoint_path}")

    def _sanitize_text(self, text):
        """Sanitizes text to only contain characters supported by the model's alphabet."""
        if not hasattr(self.dataset, 'alphabet'):
            return text
        alphabet_set = set(self.dataset.alphabet)
        mapping = {
            '!': '.',
            '?': '.',
            '"': "'",
            ';': ',',
            ':': '.',
            '[': '(',
            ']': ')',
            '{': '(',
            '}': ')',
            '_': '-',
            '—': '-',
            '–': '-',
        }
        chars = []
        for ch in text:
            if ch == ' ':
                chars.append(' ')
            elif ch in alphabet_set:
                chars.append(ch)
            elif ch in mapping and mapping[ch] in alphabet_set:
                chars.append(mapping[ch])

        cleaned = " ".join("".join(chars).split())
        return cleaned if cleaned else "a"

    def synthesize(self, text, style=107, beautify=True, seq_len=None,
                   eoc_threshold=0.05, cursive_threshold=0.005):
        """
        Synthesizes handwriting for the given text.

        Args:
            text (str): Sentence or words to generate.
            style (int or None): Reference style index (e.g., 107, 226, 696).
                                 Pass None or -1 for random unbiased handwriting style.
            beautify (bool): Applies mean-sampling filter for smooth handwriting (default True).
            seq_len (int, optional): Max stroke step length (auto-computed by default).
            eoc_threshold (float): End-of-character probability threshold.
            cursive_threshold (float): Cursive segment threshold.

        Returns:
            SynthesisResult: Object with .save(), .save_svg(), .save_png(), .to_image(), and .strokes.
        """
        if not self.is_loaded:
            raise RuntimeError("DeepWriter model is unloaded. Call load_model() or instantiate a new DeepWriter.")

        clean_text = self._sanitize_text(text)

        if seq_len is None:
            seq_len = max(600, len(clean_text) * 45)

        keyword_args = {
            'conditional_inputs': clean_text,
            'eoc_threshold': eoc_threshold,
            'cursive_threshold': cursive_threshold,
            'use_sample_mean': beautify
        }

        with self.graph.as_default():
            if style is not None and style >= 0:
                _, stroke_model_input, _ = self.dataset.fetch_sample(style)
                inference_results = self.inference_model.reconstruct_given_sample(
                    session=self.sess, inputs=stroke_model_input
                )
                results = self.sampling_model.sample_biased(
                    session=self.sess,
                    seq_len=seq_len,
                    prev_state=inference_results[0]['state'],
                    prev_sample=None,
                    **keyword_args
                )
            else:
                results = self.sampling_model.sample_unbiased(
                    session=self.sess,
                    seq_len=seq_len,
                    **keyword_args
                )

        synthetic_strokes = self.dataset.undo_normalization(
            results[0]['output_sample'][0], detrend_sample=False
        )

        return SynthesisResult(
            strokes=synthetic_strokes,
            text=text,
            style_id=style,
            beautified=beautify
        )

    def list_available_styles(self):
        """Returns list of style sample indices available in the loaded dataset."""
        if not self.is_loaded or self.dataset is None:
            return []
        return list(range(len(self.dataset.samples)))

    def unload(self):
        """
        Completely releases TensorFlow session, computational graph, and memory.
        """
        if not self.is_loaded:
            return

        try:
            if self.sess is not None:
                self.sess.close()
        except Exception:
            pass

        self.sess = None
        self.graph = None
        self.inference_model = None
        self.sampling_model = None
        self.dataset = None
        self.is_loaded = False

        # Reset global default graph and trigger Python garbage collector
        tf.reset_default_graph()
        gc.collect()

        if self.verbose:
            print("[DeepWriter] Model session and graph unloaded from memory.")

    def close(self):
        """Alias for unload()."""
        self.unload()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()


def load_model(model_dir=None, data_file=None, verbose=False):
    """
    Convenience factory to initialize and load DeepWriter.

    Example:
        with deepwriting.load_model() as writer:
            result = writer.synthesize("Hello world", style=107)
            result.save("out.png")
    """
    return DeepWriter(model_dir=model_dir, data_file=data_file, verbose=verbose)
