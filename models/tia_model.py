import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.layers import Layer, Dense, Embedding, Lambda, Add, Softmax, Dropout
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow import nn
from tensorflow.contrib.rnn import RNNCell

from .layers import CVE, Align
from .loss import TIA_Contrastive
from utils.graph_utils import cal_laplacian, cheb_polynomial

# Global variable
previous_batch_inputs = None


class Attention(Layer):
    """Attention mechanism layer"""

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        self.W = self.add_weight(
            shape=(d, self.hidden_dim),
            initializer='glorot_uniform',
            name='att_W',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.hidden_dim,),
            initializer='zeros',
            name='att_b',
            trainable=True
        )
        self.u = self.add_weight(
            shape=(self.hidden_dim, 1),
            initializer='glorot_uniform',
            name='att_u',
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, x, mask, mask_value=-1e30):
        attn_weights = K.dot(K.tanh(K.bias_add(K.dot(x, self.W), self.b)), self.u)
        mask = K.expand_dims(mask, axis=-1)
        attn_weights = mask * attn_weights + (1 - mask) * mask_value
        attn_weights = K.softmax(attn_weights, axis=-2)
        return attn_weights

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class Transformer(Layer):
    """Transformer layer"""

    def __init__(self, num_layers=2, num_heads=8, dk=None, dv=None, dff=None, dropout=0.2):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.dff = dff
        self.dropout = dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(Transformer, self).__init__()

    def build(self, input_shape):
        d_model = input_shape.as_list()[-1]

        if self.dk is None:
            self.dk = d_model // self.num_heads
        if self.dv is None:
            self.dv = d_model // self.num_heads
        if self.dff is None:
            self.dff = 2 * d_model

        self.Wq = self.add_weight(shape=(self.num_layers, self.num_heads, d_model, self.dk), name='Wq',
                                  initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.num_layers, self.num_heads, d_model, self.dk), name='Wk',
                                  initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.num_layers, self.num_heads, d_model, self.dv), name='Wv',
                                  initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.num_layers, self.num_heads * self.dv, d_model), name='Wo',
                                  initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(self.num_layers, d_model, self.dff), name='W1',
                                  initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.num_layers, self.dff), name='b1',
                                  initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.num_layers, self.dff, d_model), name='W2',
                                  initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(self.num_layers, d_model), name='b2',
                                  initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2 * self.num_layers,), name='gamma',
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2 * self.num_layers,), name='beta',
                                    initializer='zeros', trainable=True)

        super(Transformer, self).build(input_shape)

    def call(self, x, mask, mask_value=-1e-30):
        mask = K.expand_dims(mask, axis=-2)
        for i in range(self.num_layers):
            mha_ops = []
            for j in range(self.num_heads):
                q = K.dot(x, self.Wq[i, j, :, :])
                q = K.reshape(q, (-1, q.shape[1] * q.shape[2], q.shape[3]))

                k = K.permute_dimensions(K.dot(x, self.Wk[i, j, :, :]), (0, 3, 2, 1))
                k = K.reshape(k, (-1, k.shape[1], k.shape[2] * k.shape[3]))

                v = K.dot(x, self.Wv[i, j, :, :])
                v = K.reshape(v, (-1, v.shape[1] * v.shape[2], v.shape[3]))

                A = K.batch_dot(q, k)
                mask1 = K.reshape(mask, (-1, mask.shape[2], mask.shape[1] * mask.shape[3]))
                A = mask1 * A + (1 - mask1) * mask_value

                def dropped_A():
                    dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A)) >= self.dropout), K.floatx())
                    return A * dp_mask + (1 - dp_mask) * mask_value

                A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
                A = K.softmax(A, axis=-1)
                mha_ops.append(K.batch_dot(A, v))

            conc = K.concatenate(mha_ops, axis=-1)
            conc = K.reshape(conc, (-1, 3, 2, conc.shape[2]))

            proj = K.dot(conc, self.Wo[i, :, :])

            proj = tf_utils.smart_cond(K.learning_phase(),
                                       lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),
                                       lambda: array_ops.identity(proj))
            x = x + proj

            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x * self.gamma[2 * i] + self.beta[2 * i]

            ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1[i, :, :]), self.b1[i, :])),
                                      self.W2[i, :, :]), self.b2[i, :, ])
            ffn_op = tf_utils.smart_cond(K.learning_phase(),
                                         lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),
                                         lambda: array_ops.identity(ffn_op))
            x = x + ffn_op

            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x * self.gamma[2 * i + 1] + self.beta[2 * i + 1]

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class TIACell(RNNCell):
    """TIA Cell - core model unit"""

    def __init__(self, num_units, adj, k, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):
        super(TIACell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self.adj = np.mat(adj)
        self.d = 64
        self.batch_size = 32
        self.last_inputs = ""
        self.first_batch = 1  # Fixed typo from "fist_batch"
        self.k = k

    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope='TIACell'):
        varis_f = ['CrowdQuaries', 'Precipitation', 'Visibility']
        varis_f_num = len(varis_f)
        varis_inp = inputs[:, self._nodes:self._nodes + varis_f_num]
        times_inp = inputs[:, self._nodes + varis_f_num:self._nodes + 2 * varis_f_num]
        values_inp = inputs[:, self._nodes + 2 * varis_f_num:self._nodes + 3 * varis_f_num]
        spatial_inp = inputs[:, self._nodes + 3 * varis_f_num:self._nodes + 4 * varis_f_num]
        inputs = inputs[:, :self._nodes]

        global previous_batch_inputs
        if self.first_batch == 1:
            previous_batch_inputs = inputs
            self.first_batch = 0

        # Embedding layer processing
        varis_emb = Embedding(varis_f_num + 1, self.d)(varis_inp)
        cve_units = int(np.sqrt(self.d))
        times_emb = CVE(cve_units, self.d)(times_inp)
        values_emb = CVE(cve_units, self.d)(values_inp)
        spatial_emb = Embedding(self._nodes, self.d)(spatial_inp)

        # Feature fusion
        comb_emb = Add()([varis_emb, values_emb, times_emb, spatial_emb])
        mask = Lambda(lambda x: K.clip(x, 0, 1))(varis_inp)
        cont_emb = Transformer(2, 4, dk=None, dv=None, dff=None, dropout=0.2)(comb_emb, mask=mask)
        attn_weights = Attention(2 * self.d)(cont_emb, mask=mask)
        fused_emb = Lambda(lambda x: K.sum(x[0] * x[1], axis=-2))([cont_emb, attn_weights])
        fore_op = Dense(varis_f_num)(fused_emb)
        op = Dense(1, activation='sigmoid')(fore_op)

        # Graph Laplacian processing
        lap_mx = cal_laplacian(self.adj)
        Lk = cheb_polynomial(lap_mx, self.k)
        tia_tensor = []
        new_inputs_list = []

        for id in range(self.batch_size):
            way_id_index = tf.cast(tf.gather_nd(spatial_inp, [[id, 0, 0]]), tf.int32)
            way_id_state = op[id][0]
            traffic_ones = tf.cast(tf.gather_nd(previous_batch_inputs[id], [[way_id_index]]), dtype=tf.float32)
            comb_emb = Add()([way_id_state, traffic_ones])

            updated_inputs = tf.tensor_scatter_nd_update(previous_batch_inputs[id], [[way_id_index]], comb_emb)
            new_inputs = tf.tensor_scatter_nd_update(inputs[id], [[way_id_index]], comb_emb)
            new_inputs_list.append(new_inputs)
            updated_inputs = tf.transpose(updated_inputs, perm=[1, 0])
            updated_inputs = tf.expand_dims(updated_inputs, axis=0)

            for i in range(Lk.shape[0]):
                D = tf.reduce_sum(Lk[i], axis=1)
                D_sqrt_inv = tf.pow(D, -0.5)
                D_sqrt_inv = tf.where(tf.math.is_inf(D_sqrt_inv), tf.zeros_like(D_sqrt_inv), D_sqrt_inv)
                D_sqrt_inv = tf.where(tf.math.is_nan(D_sqrt_inv), tf.zeros_like(D_sqrt_inv), D_sqrt_inv)
                D_sqrt_inv = tf.linalg.diag(D_sqrt_inv)
                Lki = tf.eye(self._nodes) - tf.cast(tf.matmul(tf.matmul(D_sqrt_inv, Lk[i]), D_sqrt_inv),
                                                    dtype=tf.float32)

                tia_tensor_one = tf.matmul(updated_inputs, Lki)
                tia_tensor.append(tia_tensor_one)

        tia_tensor = tf.reshape(tia_tensor, (self.k, self.batch_size, self._nodes, 2))
        l2_list = []
        for i in range(self.k):
            score = TIA_Contrastive(self.batch_size, 6, self.batch_size, 0.5)(tia_tensor[i], inputs)
            l2_list.append(score)
        min_index = tf.argmin(l2_list)
        previous_batch_inputs = inputs

        L = tf.cast(tf.gather_nd(Lk, [min_index]), dtype=tf.float32)
        D = tf.reduce_sum(L, axis=1)
        D_sqrt_inv = tf.pow(D, -0.5)
        D_sqrt_inv = tf.where(tf.math.is_inf(D_sqrt_inv), tf.zeros_like(D_sqrt_inv), D_sqrt_inv)
        D_sqrt_inv = tf.where(tf.math.is_nan(D_sqrt_inv), tf.zeros_like(D_sqrt_inv), D_sqrt_inv)
        D_sqrt_inv = tf.linalg.diag(D_sqrt_inv)
        self.L = tf.eye(self._nodes) - tf.matmul(tf.matmul(D_sqrt_inv, L), D_sqrt_inv)
        inputs = new_inputs_list

        # GRU gating mechanism
        with tf.variable_scope(scope):
            with tf.variable_scope("gates"):
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, min_index=min_index, scope=scope, k=self.k))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, min_index=min_index, scope=scope, k=self.k))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def _gc(self, inputs, state, output_size, bias=0.0, min_index=None, scope=None, k=8):
        state = tf.reshape(state, (-1, self._nodes, self._units))
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2].value
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            x1 = tf.matmul(self.L, x0)
            x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
            x = tf.transpose(x, perm=[2, 0, 1])
            x = tf.reshape(x, shape=[-1, input_size])

            weights = tf.get_variable(
                'weights', [k, input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            weights = tf.cast(tf.gather_nd(weights, [[min_index]]), dtype=tf.float32)
            x = tf.matmul(x, weights)
            biases = tf.get_variable(
                "biases", [k, output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
            biases = tf.cast(tf.gather_nd(biases, [[min_index]]), dtype=tf.float32)
            biases = tf.reshape(biases, shape=[output_size])
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x


class SpatioConvLayer(Layer):
    """Spatial convolution layer"""

    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = tf.Variable(
            tf.random.uniform((c_in, c_out), -math.sqrt(5), math.sqrt(5)))
        self.b = tf.Variable(tf.random.uniform((1, c_out), 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        initializer_theta = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                  distribution='truncated_normal')
        self.theta.assign(initializer_theta(self.theta.shape))
        initializer_b = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='uniform')
        fan_in_b = self.b.shape.as_list()[1]
        bound = 1 / math.sqrt(fan_in_b)
        self.b.assign(tf.random.uniform(self.b.shape, minval=-bound, maxval=bound))

    def call(self, inputs, extra_param=None):
        x_c = tf.einsum("knm,bn->bm", extra_param, inputs)
        x_gc = tf.einsum("io,bm->bo", self.theta, x_c) + self.b
        return tf.nn.relu(x_gc)


def motif_gcn(inputs, _weights, _biases, num_nodes, units, output_dim, adj, k):
    """Motif Graph Convolutional Network"""
    cell_1 = TIACell(units, adj, k, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    inputs = tf.unstack(inputs, axis=1)
    outputs, states = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)

    motifs = []
    for output in outputs:
        reshaped_output = tf.reshape(output, shape=[-1, num_nodes, units])
        reshaped_output = tf.reshape(reshaped_output, shape=[-1, units])
        motifs.append(reshaped_output)

    last_output = motifs[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, output_dim])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])

    return output, motifs, states