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
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
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

    def __init__(self, cells, word_dropout=0.0, unk_index=0, embedding=None, softmax_w=None,
                 softmax_b=None, return_states=False, outputs_are_states=True, softmax_top_k=-1):
        """Create a RNN cell composed sequentially of a number of RNNCells. If embedding is not
           None, the output of the previous timestep is used for the current time step using the
           softmax variables.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self.cells = cells
        self.word_dropout = word_dropout
        self.unk_index = unk_index
        if not (embedding is None and softmax_w is None and softmax_b is None) and \
           not (embedding is not None and softmax_w is not None and softmax_b is not None):
            raise ValueError('Embedding and softmax variables have to all be None or all not None.')
        self.embedding = embedding
        self.softmax_w = softmax_w
        self.softmax_b = softmax_b
        self.return_states = return_states
        self.outputs_are_states = outputs_are_states  # should be true for GRUs
        self.softmax_top_k = softmax_top_k
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

    def expected_embedding(self, logits, prediction):
        """Use the current logits to return the embedding for the next timestep input."""
        if self.softmax_top_k == 1:
            with tf.device('/cpu:0'):
                embeddings = tf.nn.embedding_lookup(self.embedding, prediction,
                                                    name='rnn_embedding_k1')
            return embeddings
        else:
            sm = tf.nn.softmax(logits, name='Softmax')
            if self.softmax_top_k > 0:
                values, indices = tf.nn.top_k(sm, k=self.softmax_top_k, sorted=False)
                values /= tf.reduce_sum(values, -1, keep_dims=True)  # values is now a valid distrib
                with tf.device('/cpu:0'):
                    embeddings = tf.nn.embedding_lookup(self.embedding, indices,
                                                        name='rnn_embedding')
                    # rescaled embeddings by probs
                    embeddings = embeddings * tf.expand_dims(values, -1)
                    embeddings = tf.reduce_sum(embeddings, -2)  # expected embedding
                return embeddings
            else:
                return tf.matmul(sm, self.embedding)  # expectation over entire distribution

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
                prediction = tf.argmax(logits, 1)
                unknowns = tf.constant(self.unk_index, shape=prediction.get_shape(), dtype=tf.int64)
                rand = tf.random_uniform(prediction.get_shape(), minval=0, maxval=1)
                # prediction_dropped only works (and makes sense) if softmax_top_k = 1
                prediction_dropped = tf.select(tf.less(rand, self.word_dropout), unknowns,
                                               prediction)
                new_states.append(self.expected_embedding(logits, prediction_dropped))
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
