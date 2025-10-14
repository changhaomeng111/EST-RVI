import tensorflow as tf
from tensorflow.keras.layers import Dense


class TIA_Contrastive(tf.keras.Model):
    """TIA Contrastive Learning Loss"""

    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(TIA_Contrastive, self).__init__()
        self.l2norm = lambda x: tf.math.l2_normalize(x, axis=1)
        self.prototypes = Dense(units=nmb_prototype, input_shape=(c_in,), use_bias=False)
        self.nmb_prototype = nmb_prototype
        self.batch_size = batch_size
        self.tau = tau
        self.c_in = c_in
        self.weights_init()

    def weights_init(self):
        """Initialize weights for the prototype layer"""
        self.prototypes.build((None, self.c_in))
        initializer = tf.keras.initializers.glorot_uniform()
        self.prototypes.kernel.assign(initializer(self.prototypes.kernel.shape))

    def sinkhorn(self, out, epsilon=0.05, sinkhorn_iterations=3):
        """Sinkhorn algorithm for distribution alignment"""
        Q = tf.exp(out / epsilon)
        B = self.batch_size
        K = tf.cast(out.shape[1], tf.float32)

        sum_Q = tf.reduce_sum(Q)
        Q /= sum_Q
        for it in range(sinkhorn_iterations):
            # Ensure numerical stability
            Q = tf.where(tf.math.is_nan(Q), tf.zeros_like(Q), Q)
            Q = tf.where(tf.math.is_inf(Q), tf.zeros_like(Q), Q)

            Q /= tf.reduce_sum(Q, axis=1, keepdims=True) + 1e-12
            Q /= K
            Q /= tf.reduce_sum(Q, axis=0, keepdims=True) + 1e-12
            Q /= B

        Q *= B
        return tf.transpose(Q)

    def clone_weights(self):
        """Clone weights - keep the original logic unchanged"""
        weights = [tf.identity(w) for w in self.prototypes.get_weights()]
        return weights

    def call(self, z1, z2):
        """Forward propagation - keep the original logic unchanged"""
        # Process the first view
        z1 = tf.reshape(z1, (-1, self.c_in))
        z1 = self.l2norm(z1)
        zc1 = self.prototypes(z1)

        # Process the second view
        z2_reshaped = tf.reshape(z2, (-1, self.c_in))
        z2_normalized = self.l2norm(z2_reshaped)
        zc2 = self.prototypes(z2_normalized)

        # Compute Sinkhorn distributions
        q1 = self.sinkhorn(tf.stop_gradient(zc1))
        q2 = self.sinkhorn(tf.stop_gradient(zc2))
        q1 = tf.reshape(q1, shape=[-1, q1.shape[0]])
        q2 = tf.reshape(q2, shape=[-1, q2.shape[0]])

        # Compute contrastive loss
        l1_z = tf.nn.log_softmax(zc2 / self.tau, axis=1)
        l1_z = tf.reduce_sum(q1 * l1_z, axis=1)
        l1 = -tf.reduce_mean(l1_z)

        l2_z = tf.nn.log_softmax(zc1 / self.tau, axis=1)
        l2 = -tf.reduce_mean(tf.reduce_sum(q2 * l2_z, axis=1))

        return l1 + l2

    def get_config(self):
        """Get configuration - for model saving"""
        config = super(TIA_Contrastive, self).get_config()
        config.update({
            'c_in': self.c_in,
            'nmb_prototype': self.nmb_prototype,
            'batch_size': self.batch_size,
            'tau': self.tau
        })
        return config