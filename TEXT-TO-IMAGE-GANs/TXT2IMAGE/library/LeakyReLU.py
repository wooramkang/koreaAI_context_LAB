from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Concatenate, Dropout
from keras.layers import Activation
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K

class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit.
    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: float >= 0. Negative slope coefficient.
    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](
           https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
    """

    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(LeakyReLU, self).get_config()
        return dict( list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape