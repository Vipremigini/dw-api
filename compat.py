import sys
import imageio.v2 as imageio
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import mpl_toolkits.axes_grid1.inset_locator as inset_locator

# 1. Compatibility for scipy.misc.imsave
def _imsave(path, arr):
    imageio.imwrite(path, arr)

scipy.misc.imsave = _imsave

# 2. Compatibility for matplotlib InsetPosition
class InsetPosition(object):
    def __init__(self, parent, lbwh):
        self.parent = parent
        self.lbwh = lbwh
    def __call__(self, ax, renderer=None):
        bbox = self.parent.get_position(original=False)
        return Bbox.from_bounds(
            bbox.x0 + self.lbwh[0] * bbox.width,
            bbox.y0 + self.lbwh[1] * bbox.height,
            self.lbwh[2] * bbox.width,
            self.lbwh[3] * bbox.height
        )

inset_locator.InsetPosition = InsetPosition

# 3. Compatibility for TensorFlow 1.x under TensorFlow 2.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class _ContribRNN:
    RNNCell = tf.nn.rnn_cell.RNNCell
    BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
    LSTMCell = tf.nn.rnn_cell.LSTMCell
    DropoutWrapper = tf.nn.rnn_cell.DropoutWrapper
    MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell

class _Contrib:
    rnn = _ContribRNN
    distributions = tf.distributions

tf.contrib = _Contrib()
sys.modules['tensorflow'] = tf
