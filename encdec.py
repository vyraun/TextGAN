import tensorflow as tf

from config import cfg
import rnncell
import utils


class EncoderDecoderModel(object):

    '''The encoder-decoder adversarial model.'''

    def __init__(self, vocab, training, mle_mode, mle_reuse, gan_reuse, g_optimizer=None,
                 d_optimizer=None, mle_generator=False):
        self.vocab = vocab
        self.training = training
        self.mle_mode = mle_mode
        self.mle_reuse = mle_reuse
        self.gan_reuse = gan_reuse
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.reuse = mle_reuse or gan_reuse  # for common variables

        self.embedding = self.word_embedding_matrix()
        self.softmax_w, self.softmax_b = self.softmax_variables()

        if mle_mode:
            # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
            self.data = tf.placeholder(tf.int32, [cfg.batch_size, None], name='data')
            # sentence lengths
            self.lengths = tf.placeholder(tf.int32, [cfg.batch_size], name='lengths')
            # sentences with word dropout
            self.data_dropped = tf.placeholder(tf.int32, [cfg.batch_size, None],
                                               name='data_dropped')

            embs = self.word_embeddings(self.data)
            embs_dropped = self.word_embeddings(self.data_dropped)
            self.latent = self.encoder(embs_dropped[:, 1:, :])
        else:
            # only the first timestep input will actually be considered
            embs = self.word_embeddings(tf.constant(vocab.sos_index, shape=[cfg.batch_size, 1]))
            # so the rest can be zeros
            embs = tf.concat(1, [embs, tf.zeros([cfg.batch_size, cfg.gen_sent_length - 1,
                                                 cfg.emb_size])])
            if mle_generator:
                self.latent = tf.placeholder(tf.float32, [cfg.batch_size,
                                                          cfg.num_layers * cfg.hidden_size],
                                             name='mle_generator_input')
            else:
                self.rand_input = tf.placeholder(tf.float32,
                                                 [cfg.batch_size, cfg.hidden_size],
                                                 name='gan_random_input')
                self.latent = self.generate_latent(self.rand_input)
        output, states, self.generated = self.decoder(embs, self.latent)

        if not mle_mode:
            self.lengths = self.compute_lengths()
        if cfg.d_rnn:
            d_out = self.discriminator_rnn(states)
        else:
            d_out = self.discriminator_finalstate(states)
        if mle_mode:
            # shift left the input to get the targets
            targets = tf.concat(1, [self.data[:, 1:], tf.zeros([cfg.batch_size, 1], tf.int32)])
            mle_loss = self.mle_loss(output, targets)
            self.nll = tf.reduce_sum(mle_loss) / cfg.batch_size
            self.mle_cost = self.nll
            gan_loss = self.gan_loss(d_out, 1)
            self.gan_cost = tf.reduce_sum(gan_loss) / cfg.batch_size
            if training:
                self.mle_train_op = self.train_mle(self.mle_cost)
                self.mle_encoder_train_op = self.train_mle_encoder(self.mle_cost)
                self.d_train_op = self.train_d(self.gan_cost)
            else:
                self.mle_train_op = tf.no_op()
                self.mle_encoder_train_op = tf.no_op()
                self.d_train_op = tf.no_op()
        else:
            gan_loss = self.gan_loss(d_out, 0)
            self.gan_cost = tf.reduce_sum(gan_loss) / cfg.batch_size
            if training:
                self.d_train_op = self.train_d(self.gan_cost)
                self.g_train_op = self.train_g(-self.gan_cost)
            else:
                self.d_train_op = tf.no_op()
                self.g_train_op = tf.no_op()

    def rnn_cell(self, num_layers, hidden_size, latent=None, embedding=None, softmax_w=None,
                 softmax_b=None, return_states=False):
        '''Return a multi-layer RNN cell.'''
        softmax_top_k = cfg.generator_top_k
        if softmax_top_k > 0 and len(self.vocab.vocab) <= softmax_top_k:
            softmax_top_k = -1
        return rnncell.MultiRNNCell([rnncell.GRUCell(hidden_size, latent=latent)
                                     for _ in xrange(num_layers)],
                                    embedding=embedding, softmax_w=softmax_w, softmax_b=softmax_b,
                                    return_states=return_states, softmax_top_k=softmax_top_k)

    def word_embedding_matrix(self):
        '''Define the word embedding matrix.'''
        with tf.device('/cpu:0') and tf.variable_scope("Embeddings", reuse=self.reuse):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           cfg.emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
        return embedding

    def softmax_variables(self):
        '''Define the softmax weight and bias variables.'''
        with tf.variable_scope("MLE_Softmax", reuse=self.reuse):
            softmax_w = tf.get_variable("W", [len(self.vocab.vocab), cfg.hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("b", [len(self.vocab.vocab)],
                                        initializer=tf.zeros_initializer)
        return softmax_w, softmax_b

    def word_embeddings(self, inputs):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0'):
            embeds = tf.nn.embedding_lookup(self.embedding, inputs, name='word_embedding_lookup')
        return embeds

    def generate_latent(self, rand_input):
        '''Transform a sample from the normal distribution to a sample from the latent
           representation distribution.'''
        with tf.variable_scope("Transform_Latent", reuse=self.gan_reuse):
            rand_input = utils.highway(rand_input, layer_size=1)
            latent = utils.linear(rand_input, cfg.num_layers * cfg.hidden_size,
                                  True)
        return tf.nn.tanh(latent)

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation in MLE mode.'''
        with tf.variable_scope("Encoder", reuse=self.mle_reuse):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(cfg.num_layers, cfg.hidden_size), inputs,
                                         sequence_length=self.lengths-1, swap_memory=True,
                                         dtype=tf.float32)
            latent = utils.highway(state)
        return latent

    def decoder(self, inputs, latent):
        '''Use the latent representation and word inputs to predict next words.'''
        with tf.variable_scope("Decoder", reuse=self.reuse):
            if self.mle_mode:
                outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(cfg.num_layers, cfg.hidden_size,
                                                             latent, return_states=True), inputs,
                                               sequence_length=self.lengths-1, swap_memory=True,
                                               dtype=tf.float32)
            else:
                outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(cfg.num_layers, cfg.hidden_size,
                                                             latent, self.embedding, self.softmax_w,
                                                             self.softmax_b, return_states=True),
                                               inputs, swap_memory=True, dtype=tf.float32)
            output = tf.slice(outputs, [0, 0, 0], [-1, -1, cfg.hidden_size])
            if self.mle_mode:
                generated = None
                skip = 0
            else:
                words = tf.squeeze(tf.cast(tf.slice(outputs, [0, 0, cfg.hidden_size],
                                                    [-1, cfg.gen_sent_length - 1, 1]),
                                           tf.int32), [-1])
                generated = tf.stop_gradient(tf.concat(1, [words, tf.constant(self.vocab.eos_index,
                                                               shape=[cfg.batch_size, 1])]))
                skip = 1
            states = tf.slice(outputs, [0, 0, cfg.hidden_size + skip], [-1, -1, -1])
            # for GRU, we skipped the last layer states because they're the outputs
            states = tf.concat(2, [states, output])
        return output, states, generated

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        mask = tf.cast(tf.greater(targets, 0, name='targets_mask'), tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, cfg.hidden_size])
        if self.training and cfg.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, output, targets,
                                              cfg.softmax_samples, len(self.vocab.vocab))
            loss *= mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(self.softmax_w),
                                              name='softmax_transform_mle'), self.softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(targets, [-1])],
                                                          [tf.reshape(mask, [-1])])
        return tf.reshape(loss, [cfg.batch_size, -1])

    def compute_lengths(self):
        eos_locs = tf.where(tf.equal(self.generated, self.vocab.eos_index))
        counts = tf.unique_with_counts(eos_locs[:, 0])[2]
        meta_indices = tf.expand_dims(tf.cumsum(counts, exclusive=True), -1)
        gather_indices = tf.concat(1, [meta_indices, tf.ones_like(meta_indices)])
        indices = tf.cast(tf.gather_nd(eos_locs, gather_indices), tf.int32)
        return indices + 2  # +1 for the assumed <sos> in the beginning

    def discriminator_rnn(self, states):
        '''Recurrent discriminator that operates on the sequence of states of the sentences.'''
        with tf.variable_scope("Discriminator", reuse=self.reuse):
            if cfg.d_rnn_bidirect:
                hidden_size = cfg.hidden_size // 2
                lengths = tf.cast(self.lengths, tf.int64)  # workaround to make the following work
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell(cfg.d_num_layers,
                                                                           hidden_size,
                                                                           return_states=True),
                                                             self.rnn_cell(cfg.d_num_layers,
                                                                           hidden_size,
                                                                           return_states=True),
                                                             states, sequence_length=lengths-1,
                                                             swap_memory=True, dtype=tf.float32)
            else:
                hidden_size = cfg.hidden_size
                outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(cfg.d_num_layers, hidden_size,
                                                             return_states=True), states,
                                               sequence_length=self.lengths-1, swap_memory=True,
                                               dtype=tf.float32)
                outputs = (outputs,)  # to match bidirectional RNN's output format
            d_states = []
            for out in outputs:
                output = tf.slice(out, [0, 0, 0], [-1, -1, hidden_size])
                dir_states = tf.slice(out, [0, 0, hidden_size], [-1, -1, -1])
                # for GRU, we skipped the last layer states because they're the outputs
                d_states.append(tf.concat(2, [dir_states, output]))
        return self._discriminator_conv(tf.concat(2, d_states))

    def _discriminator_conv(self, states):
        '''Convolve output of bidirectional RNN and predict the discriminator label.'''
        with tf.variable_scope("Discriminator", reuse=self.reuse):
            W_conv = tf.get_variable('W_conv', [cfg.d_conv_window, 1, states.get_shape()[2],
                                                cfg.hidden_size // cfg.d_conv_window],
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_conv = tf.get_variable('b_conv', [cfg.hidden_size // cfg.d_conv_window],
                                     initializer=tf.constant_initializer(0.0))
            states = tf.expand_dims(states, 2)
            conv = tf.nn.conv2d(states, W_conv, strides=[1, 1, 1, 1], padding='SAME')
            conv_out = tf.reshape(conv, [cfg.batch_size, -1, cfg.hidden_size // cfg.d_conv_window])
            conv_out = tf.nn.elu(tf.nn.bias_add(conv_out, b_conv))
            reduced = tf.reduce_mean(conv_out, [1])
            lin_latent = tf.nn.elu(utils.linear(self.latent, cfg.hidden_size // cfg.d_conv_window,
                                                True, 0.0, scope='lin_latent'))
            reduced = tf.concat(1, [lin_latent, reduced])
            output = utils.linear(reduced, 1, True, 0.0, scope='discriminator_output')
        return output

    def discriminator_finalstate(self, states):
        '''Discriminator that operates on the final states of the sentences.'''
        with tf.variable_scope("Discriminator", reuse=self.reuse):
            # indices = lengths - 2, since the generated output skips <sos>
            final_states = utils.rowwise_lookup(states, self.lengths - 2)
            combined = tf.concat(1, [self.latent, final_states])
            lin1 = tf.nn.elu(utils.linear(combined, cfg.hidden_size, True, 0.0,
                                          scope='discriminator_lin1'))
            output = utils.linear(lin1, 1, True, 0.0, scope='discriminator_output')
        return output

    def gan_loss(self, d_out, label):
        '''Return the discriminator loss according to the label. Put no variables here.'''
        return tf.nn.sigmoid_cross_entropy_with_logits(d_out, tf.constant(label, dtype=tf.float32,
                                                                          shape=d_out.get_shape()))

    def _train(self, cost, scope, optimizer):
        '''Generic training helper'''
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads = tf.gradients(cost, tvars)
        if cfg.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, cfg.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars))

    def train_mle(self, cost):
        '''Training op for MLE mode.'''
        return self._train(cost, '.*/(Embeddings|Encoder|Decoder|MLE_Softmax)', self.g_optimizer)

    def train_mle_encoder(self, cost):
        '''Encoder-only training op for MLE mode.'''
        return self._train(cost, '.*/Encoder', self.g_optimizer)

    def train_d(self, cost):
        '''Training op for GAN mode, discriminator.'''
        return self._train(cost, '.*/Discriminator', self.d_optimizer)

    def train_g(self, cost):
        '''Training op for GAN mode, generator.'''
        # don't update embeddings, just update the generated distributions
        return self._train(cost, '.*/(Transform_Latent|Decoder)', self.g_optimizer)
