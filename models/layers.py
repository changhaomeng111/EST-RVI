import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras import backend as K

class CVE(Layer):
    """Contextual Variable Encoding"""
    def __init__(self, hid_units, output_dim):
        super(CVE, self).__init__()
        self.hid_units = hid_units
        self.output_dim = output_dim

    def build(self, input_shape):
        self.W1 = self.add_weight(
            name='CVE_W1',
            shape=(1, self.hid_units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b1 = self.add_weight(
            name='CVE_b1',
            shape=(self.hid_units,),
            initializer='zeros',
            trainable=True
        )
        self.W2 = self.add_weight(
            name='CVE_W2',
            shape=(self.hid_units, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super(CVE, self).build(input_shape)

    def call(self, x):
        x = K.expand_dims(x, axis=-1)
        x = K.dot(K.tanh(K.bias_add(K.dot(x, self.W1), self.b1)), self.W2)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

class Align(Layer):
    """Spatial-Temporal Feature Alignment Layer"""
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = Dense(c_out, use_bias=False)

    def call(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, self.c_out - self.c_in]])
            return tf.pad(x, paddings)
        return x