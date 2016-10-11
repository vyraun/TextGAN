import tensorflow as tf

import utils


class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
       This variant can be conditioned on a provided latent variable.
       Based on the code from TensorFlow."""

    def __init__(self, num_units, latent=None, activation=tf.nn.tanh):
        self.num_units = num_units
        self.latent = latent
        self.activation = activation

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


class MultiRNNCell(tf.nn.rnn_cell.RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
            cells: list of RNNCells that will be composed in this order.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self.cells = cells

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self.cells)

    @property
    def output_size(self):
        return self.cells[-1].output_size

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self.cells):
                with tf.variable_scope("Cell%d" % i):
                    if not tf.nn.nest.is_sequence(state):
                        raise ValueError(
                                       "Expected state to be a tuple of length %d, but received: %s"
                                       % (len(self.state_size), state))
                    cur_state = state[i]
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states

