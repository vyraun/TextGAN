import tensorflow as tf

import utils


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
        self.outputs_are_states = outputs_are_states  # should be true for GRUs
        if embedding is not None:
            self.emb_size = embedding.get_shape()[1]
        else:
            self.emb_size = 0

    @property
    def state_size(self):
        sizes = [cell.state_size for cell in self.cells]
        if self.emb_size:
            sizes.extend([self.emb_size, 1])
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
        if self.emb_size:
            size += 1  # for the current timestep prediction
        return size

    def initial_state(self, initial):
        '''Generate the required initial state from $initial.'''
        if self.emb_size:
            initial.append(tf.zeros([initial[0].get_shape()[0], self.emb_size]))
            initial.append(tf.zeros([initial[0].get_shape()[0], 1]))
        return tuple(initial)

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
            if self.embedding is not None:
                cur_inp = tf.select(tf.greater(state[-1][:, 0], 0.5), state[-2], inputs)
            else:
                cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self.cells):
                with tf.variable_scope("Layer%d" % i):
                    if not tf.nn.nest.is_sequence(state):
                        raise ValueError("Expected state to be a tuple of length %d, but received: "
                                         "%s" % (len(self.state_size), state))
                    cur_state = state[i]
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)
            if self.embedding is not None:
                logits = tf.nn.bias_add(tf.matmul(cur_inp, tf.transpose(self.softmax_w),
                                                  name='Softmax_transform'),
                                        self.softmax_b)
                logits = tf.nn.log_softmax(logits)
                dist = tf.contrib.distributions.Categorical(logits)
                prediction = tf.cast(dist.sample(), tf.int64)
                with tf.device('/cpu:0'):
                    embeddings = tf.nn.embedding_lookup(self.embedding, prediction,
                                                        name='rnn_embedding_k1')
                new_states.append(embeddings)
                new_states.append(tf.ones([inputs.get_shape()[0], 1]))  # we have valid prev input
        if self.return_states:
            output = [cur_inp]
            if self.embedding is not None:
                skip = 2
                output.append(tf.cast(tf.expand_dims(prediction, -1), tf.float32))
            else:
                skip = 0
            if self.outputs_are_states:  # skip the last layer states, since they're outputs
                skip += 1
            return tf.concat(1, output + new_states[:-skip]), tuple(new_states)
        else:
            return cur_inp, tuple(new_states)
