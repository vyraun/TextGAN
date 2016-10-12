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

    def __init__(self, cells, embedding=None, softmax_w=None, softmax_b=None, return_states=False,
                 outputs_are_states=True):
        """Create a RNN cell composed sequentially of a number of RNNCells. If embedding is not
           None, the output of the previous timestep is used for the current time step using the
           softmax variables.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self.cells = cells
        if not (embedding is None and softmax_w is None and softmax_b is None) and \
           not (embedding is not None and softmax_w is not None and softmax_b is not None):
            raise ValueError('Embedding and softmax variables have to all be None or all not None.')
        self.embedding = embedding
        self.softmax_w = softmax_w
        self.softmax_b = softmax_b
        self.return_states = return_states
        self.outputs_are_states = outputs_are_states # should be true for GRUs
        if embedding is not None:
            self.word_emb_size = embedding.get_shape()[1]
        else:
            self.word_emb_size = 0

    @property
    def state_size(self):
        sizes = [cell.state_size for cell in self.cells]
        if self.word_emb_size:
            sizes.extend([self.word_emb_size, 1])
        return tuple(sizes)

    @property
    def output_size(self):
        size = self.cells[-1].output_size
        if self.outputs_are_states:
            skip = 1
        else:
            skip = 0
        if self.return_states:
            size += sum(cell.state_size for cell in self.cells[:-skip])
        return size

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
            cur_state_pos = 0
            if self.embedding is not None:
                cur_inp = tf.select(tf.greater(state[-1][:,0], 0.5), state[-2], inputs)
            else:
                cur_inp = inputs
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self.cells):
                with tf.variable_scope("Layer%d" % i):
                    if not tf.nn.nest.is_sequence(state):
                        raise ValueError(
                                       "Expected state to be a tuple of length %d, but received: %s"
                                       % (len(self.state_size), state))
                    cur_state = state[i]
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)
            if self.embedding is not None:
                logits = tf.nn.bias_add(tf.matmul(cur_inp, tf.transpose(self.softmax_w),
                                                  name='Softmax_transform'),
                                        self.softmax_b) # TODO do a sparse approximation?
                sm = tf.nn.softmax(logits, name='Softmax')
                new_states.append(tf.matmul(sm, self.embedding))
                new_states.append(tf.ones([sm.get_shape()[0], 1]))
        if self.return_states:
            if self.embedding is not None:
                skip = 2
            else:
                skip = 0
            if self.outputs_are_states: # skip the last layer states, since they're outputs
                skip += 1
            return tf.concat(1, [cur_inp] + new_states[:-skip]), tuple(new_states)
        else:
            return cur_inp, tuple(new_states)

