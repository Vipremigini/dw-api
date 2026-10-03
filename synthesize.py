#!/usr/bin/env python
"""
DeepWriting Handwriting Synthesis Tool
Generates synthetic handwriting in vector (SVG) and raster (PNG) formats using the pretrained C-VRNN model.
"""
import os
import sys

# Ensure source directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'source')))

import compat
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tf_dataset_hw import HandWritingDatasetConditionalTF, HandWritingDatasetConditional, HandWritingDataset
from tf_models import VRNNGMM
from tf_models_hw import HandwritingVRNNGmmModel, HandwritingVRNNModel
import visualize_hw as visualize


def render_strokes_png(strokes, factor=0.001, output_path='output.png', linewidth=2.0, color='black'):
    """Renders stroke offsets to a high-resolution PNG using matplotlib."""
    abs_x = 0
    abs_y = 0
    lines = []
    current_line = []

    for idx in range(len(strokes)):
        dx = float(strokes[idx, 0]) / factor
        dy = float(strokes[idx, 1]) / factor
        abs_x += dx
        abs_y += dy
        current_line.append((abs_x, -abs_y))  # Invert Y so handwriting is right-side up

        if strokes[idx, 2] == 1:  # Pen up
            if len(current_line) > 1:
                lines.append(np.array(current_line))
            current_line = []

    if len(current_line) > 1:
        lines.append(np.array(current_line))

    if not lines:
        print("Warning: No visible strokes to render.")
        return

    # Calculate bounding box
    all_pts = np.concatenate(lines, axis=0)
    width = all_pts[:, 0].max() - all_pts[:, 0].min()
    height = all_pts[:, 1].max() - all_pts[:, 1].min()

    aspect_ratio = max(width / max(height, 1.0), 1.0)
    fig_w = min(max(6.0, aspect_ratio * 1.8), 24.0)
    fig_h = max(2.5, fig_w / aspect_ratio)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=linewidth, solid_capstyle='round')

    ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)


def synthesize_text(model_dir, validation_data_path, text, style_id=None, beautify=True, output_prefix='synthesis'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    config['model_dir'] = model_dir
    config['checkpoint_id'] = None

    Model_cls = getattr(sys.modules[__name__], config['model_cls'])
    Dataset_cls = getattr(sys.modules[__name__], config['dataset_cls'])

    batch_size = 1
    data_sequence_length = None

    print(f"Loading dataset statistics from {validation_data_path}...")
    if issubclass(Dataset_cls, HandWritingDatasetConditional):
        dataset = Dataset_cls(validation_data_path, var_len_seq=True, use_bow_labels=config['use_bow_labels'])
    elif issubclass(Dataset_cls, HandWritingDataset):
        dataset = Dataset_cls(validation_data_path, var_len_seq=True)
    else:
        raise Exception("Unknown dataset class.")

    # Placeholders
    strokes = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(dataset.input_dims)])
    targets = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(dataset.target_dims)])
    sequence_length = tf.placeholder(tf.int32, shape=[batch_size])

    # Build inference graph (used for style conditioning)
    with tf.name_scope("validation"):
        inference_model = Model_cls(config,
                                    reuse=False,
                                    input_op=strokes,
                                    target_op=targets,
                                    input_seq_length_op=sequence_length,
                                    input_dims=dataset.input_dims,
                                    target_dims=dataset.target_dims,
                                    batch_size=batch_size,
                                    mode="validation",
                                    data_processor=dataset)
        inference_model.build_graph()

    # Build sampling graph
    with tf.name_scope("sampling"):
        sampling_model = Model_cls(config,
                                   reuse=True,
                                   input_op=strokes,
                                   target_op=None,
                                   input_seq_length_op=sequence_length,
                                   input_dims=dataset.input_dims,
                                   target_dims=dataset.target_dims,
                                   batch_size=batch_size,
                                   mode="sampling",
                                   data_processor=dataset)
        sampling_model.build_graph()

    sess = tf.Session()
    saver = tf.train.Saver()
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    print(f"Restoring checkpoint from {checkpoint_path}...")
    saver.restore(sess, checkpoint_path)

    # Sampling parameters
    seq_len = max(600, len(text) * 45)  # Dynamic sequence length based on character count
    keyword_args = {
        'conditional_inputs': text,
        'eoc_threshold': 0.05,
        'cursive_threshold': 0.005,
        'use_sample_mean': beautify
    }

    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Synthesizing text: \"{text}\"")

    if style_id is not None and style_id >= 0:
        print(f"Conditioning on reference style sample ID: {style_id}")
        _, stroke_model_input, _ = dataset.fetch_sample(style_id)
        inference_results = inference_model.reconstruct_given_sample(session=sess, inputs=stroke_model_input)

        results = sampling_model.sample_biased(session=sess,
                                              seq_len=seq_len,
                                              prev_state=inference_results[0]['state'],
                                              prev_sample=None,
                                              **keyword_args)
    else:
        print("Using unbiased random style sampling...")
        results = sampling_model.sample_unbiased(session=sess, seq_len=seq_len, **keyword_args)

    synthetic_strokes = dataset.undo_normalization(results[0]['output_sample'][0], detrend_sample=False)

    svg_path = output_prefix + '.svg'
    png_path = output_prefix + '.png'

    visualize.draw_stroke_svg(synthetic_strokes, factor=0.001, svg_filename=svg_path)
    render_strokes_png(synthetic_strokes, factor=0.001, output_path=png_path)

    print(f"Successfully saved:")
    print(f"  Vector SVG: {svg_path}")
    print(f"  Raster PNG: {png_path}")

    sess.close()
    return svg_path, png_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepWriting Handwriting Synthesis CLI")
    parser.add_argument('-t', '--text', type=str, default="DeepWriting synthesized text", help="Text to synthesize")
    parser.add_argument('-s', '--style', type=int, default=107, help="Style sample index (e.g. 107, 226, 696), or -1 for random unbiased style")
    parser.add_argument('-o', '--output', type=str, default="output/synthesis_sample", help="Output file prefix (without extension)")
    parser.add_argument('-b', '--beautify', action="store_true", default=True, help="Enable handwriting beautification (use_sample_mean)")
    parser.add_argument('--no-beautify', dest='beautify', action="store_false", help="Disable handwriting beautification")
    parser.add_argument('-m', '--model_dir', type=str, default="pretrained_models/tf-1514981744-deepwriting_synthesis_model", help="Path to pretrained model directory")
    parser.add_argument('-d', '--data_file', type=str, default="data/deepwriting_validation.npz", help="Path to validation data file")

    args = parser.parse_args()
    synthesize_text(args.model_dir, args.data_file, args.text, args.style, args.beautify, args.output)
