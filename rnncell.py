import tensorflow as tf

import utils


class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
       This variant can be conditioned on a provided latent variable.
       Based on the code from TensorFlow."""

    def __init__(self, num_units, latent=None, embedding=None, softmax_w=None, softmax_b=None,
                 activation=tf.nn.tanh):
        '''If embedding are not None, only the first timestep input is considered and the rest
           come from previous timesteps, using the softmax variables passed.''' # TODO
        self.num_units = num_units
        self.latent = latent
        self.activation = activation
        if not (embedding is not None and softmax_w is not None and softmax_b is not None) and \
           not (embedding is None and softmax_w is None and softmax_b is None):
            raise ValueError('Embedding and softmax vars have to be all None or all not None.')
        self.embedding = embedding
        self.softmax_w = softmax_w
        self.softmax_b = softmax_b

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with num_units cells."""
        with tf.variable_scope(scope or type(self).__name__): # "GRUCell"
            with tf.variable_scope("Gates"): # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                factors = [inputs, state]
                if self.latent is not None:
                    factors.append(self.latent)
                r, u = tf.split(1, 2, utils.linear(factors, 2 * self.num_units, True, 1.0))
                r, u = tf.nn.sigmoid(r), tf.nn.sigmoid(u)
            with tf.variable_scope("Candidate"):
                factors = [inputs, r * state]
                if self.latent is not None:
                    factors.append(self.latent)
                c = self.activation(utils.linear(factors, self.num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """Basic LSTM recurrent network cell (http://arxiv.org/abs/1409.2329).
       Based on the code from TensorFlow."""

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        """Initialize the basic LSTM cell.
           Args:
             num_units: int, The number of units in the LSTM cell.
             forget_bias: float, The bias added to forget gates.
             activation: Activation function of the inner states."""
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__): # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = state
            concat = utils.linear([inputs, h], 4 * self.num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)

            new_c = (c * tf.nn.sigmoid(f + self.forget_bias) + tf.nn.sigmoid(i) * \
                     self.activation(j))
            new_h = self.activation(new_c) * tf.nn.sigmoid(o)

            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state
