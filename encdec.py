import tensorflow as tf

from config import cfg
import rnncell
import utils


class EncoderDecoderModel(object):

    '''The variational encoder-decoder adversarial model.'''

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
            with tf.variable_scope("GlobalMLE", reuse=self.reuse):
                self.global_step = tf.get_variable('global_step', shape=[],
                                                   initializer=tf.zeros_initializer,
                                                   trainable=False)
            # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
            self.data = tf.placeholder(tf.int32, [cfg.batch_size, None], name='data')
            # sentence lengths
            # TODO get rid of dynamic sequence lengths -- asking for too much! these are still
            #      relevant for encoding though.
            self.lengths = tf.placeholder(tf.int32, [cfg.batch_size], name='lengths')
            # sentences with word dropout
            self.data_dropped = tf.placeholder(tf.int32, [cfg.batch_size, None],
                                               name='data_dropped')

            embs = self.word_embeddings(self.data)
            embs_dropped = self.word_embeddings(self.data_dropped)
            embs_reversed = tf.reverse_sequence(embs, self.lengths, 1)
            z_mean, z_logvar = self.encoder(embs_reversed[:, 1:, :])
            eps = tf.random_normal([cfg.batch_size, cfg.latent_size])
            self.latent = z_mean + tf.mul(tf.sqrt(tf.exp(z_logvar)), eps)
        else:
            # only the first timestep input will actually be considered
            embs_dropped = self.word_embeddings(tf.constant(vocab.sos_index,
                                                            shape=[cfg.batch_size, 1]))
            # so the rest can be zeros
            embs_dropped = tf.concat(1, [embs_dropped, tf.zeros([cfg.batch_size,
                                                                 cfg.gen_sent_length - 1,
                                                                 cfg.emb_size])])
            if mle_generator:
                self.latent = tf.placeholder(tf.float32, [cfg.batch_size, cfg.latent_size],
                                             name='mle_generator_input')
            else:
                self.latent = tf.random_normal([cfg.batch_size, cfg.latent_size],
                                               name='gan_random_input')
        output, states, self.generated = self.decoder(embs_dropped, self.latent)

        if not mle_mode:
            self.lengths = self.compute_lengths()
        if cfg.d_energy_based:
            d_out, d_latent = self.discriminator_energy(states)
        elif cfg.d_rnn:
            d_out = self.discriminator_rnn(states)
        else:
            d_out = self.discriminator_finalstate(states)
        if cfg.d_energy_based:
            # shift left the states to get the targets
            targets = tf.concat(1, [states, tf.expand_dims(d_latent, 1)])
            d_loss, g_loss = self.gan_energy_loss(d_out, targets)
            self.d_cost = tf.reduce_sum(d_loss) / cfg.batch_size
            self.g_cost = tf.reduce_sum(g_loss) / cfg.batch_size
        else:
            gan_loss = self.gan_loss(d_out)
            self.d_cost = tf.reduce_sum(gan_loss) / cfg.batch_size
            self.g_cost = -self.d_cost

        if mle_mode:
            # shift left the input to get the targets
            targets = tf.concat(1, [self.data[:, 1:], tf.zeros([cfg.batch_size, 1], tf.int32)])
            self.nll = tf.reduce_sum(self.mle_loss(output, targets)) / cfg.batch_size
            self.kld = tf.reduce_sum(self.kld_loss(z_mean, z_logvar)) / cfg.batch_size
            self.kld_weight = cfg.anneal_max * tf.sigmoid((7 / cfg.anneal_bias) *
                                                          (self.global_step - cfg.anneal_bias))
            self.mle_cost = self.nll + (self.kld_weight * self.kld)
            if training:
                self.mle_train_op = self.train_mle(self.mle_cost)
                self.mle_encoder_train_op = self.train_mle_encoder(self.mle_cost)
                self.d_train_op = self.train_d(self.d_cost)
            else:
                self.mle_train_op = tf.no_op()
                self.mle_encoder_train_op = tf.no_op()
                self.d_train_op = tf.no_op()
        else:
            if training:
                self.d_train_op = self.train_d(self.d_cost)
                self.g_train_op = self.train_g(self.g_cost)
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
                                     for _ in range(num_layers)], word_dropout=cfg.word_dropout,
                                    unk_index=self.vocab.unk_index, embedding=embedding,
                                    softmax_w=softmax_w, softmax_b=softmax_b,
                                    return_states=return_states, softmax_top_k=softmax_top_k,
                                    use_argmax=cfg.generate_argmax)

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

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation in MLE mode.'''
        with tf.variable_scope("Encoder", reuse=self.mle_reuse):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(cfg.num_layers, cfg.hidden_size), inputs,
                                         sequence_length=self.lengths-1, swap_memory=True,
                                         dtype=tf.float32)
            # TODO make the encoder a BiRNN+convnet (try VAE first)
            latent = utils.highway(state, layer_size=2)
            z_mean = utils.linear(latent, cfg.latent_size, True, 0.0, scope='Latent_mean')
            z_logvar = utils.linear(latent, cfg.latent_size, True, 0.0, scope='Latent_logvar')
        return z_mean, z_logvar

    def decoder(self, inputs, latent):
        '''Use the latent representation and word inputs to predict next words.'''
        with tf.variable_scope("Decoder", reuse=self.reuse):
            latent = utils.highway(latent, layer_size=1)
            latent = utils.linear(latent, cfg.latent_size, True, 0.0, scope='Latent_transform')
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
        present_mask = tf.greater(targets, 0, name='present_mask')
        if cfg.word_dropout > 0.99:
            # don't enfoce loss on true <unk>'s
            unk_mask = tf.not_equal(targets, self.vocab.unk_index, name='unk_mask')
            mask = tf.cast(tf.logical_and(present_mask, unk_mask), tf.float32)
        else:
            mask = tf.cast(present_mask, tf.float32)
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

    def kld_loss(self, z_mean, z_logvar):
        '''KL divergence loss.'''
        z_var = tf.exp(z_logvar)
        z_mean_sq = tf.square(z_mean)
        return 0.5 * tf.reduce_sum(z_var + z_mean_sq - 1 - z_logvar, 1)

    def compute_lengths(self):
        '''Compute sentence lengths from generated sentences.'''
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
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
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

    def discriminator_energy(self, states):
        '''An energy-based discriminator that tries to reconstruct the input states.'''
        with tf.variable_scope("Discriminator", reuse=self.reuse):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(cfg.d_num_layers, cfg.hidden_size), states,
                                         sequence_length=self.lengths-1, swap_memory=True,
                                         dtype=tf.float32, scope='discriminator_encoder')
            # TODO use BiRNN+convnet for the encoder (try VAE first)
            # this latent is of size cfg.hidden_size since it needs a lot more capacity than
            # cfg.latent_size to reproduce the hidden states
            latent = utils.highway(state, layer_size=1)
            latent = utils.linear(latent, cfg.hidden_size, True,
                                  scope='discriminator_latent_transform')
            decoder_input = tf.concat(1, [tf.zeros([cfg.batch_size, 1, cfg.hidden_size]), states])
            output, _ = tf.nn.dynamic_rnn(self.rnn_cell(cfg.d_num_layers, cfg.hidden_size, latent),
                                          decoder_input, sequence_length=self.lengths,
                                          swap_memory=True, dtype=tf.float32,
                                          scope='discriminator_decoder')
            output = tf.reshape(output, [-1, cfg.hidden_size])
            reconstructed = utils.linear(output, cfg.hidden_size, True, 0.0,
                                         scope='discriminator_reconst')
            reconstructed = tf.reshape(reconstructed, [cfg.batch_size, -1, cfg.hidden_size])
            # don't train this projection, since the model can learn to zero out ret_latent to
            # minimize the reconstruction error
            ret_latent = tf.nn.tanh(utils.linear(self.latent, cfg.hidden_size, False,
                                                 scope='discriminator_ret_latent', train=False))
        return reconstructed, ret_latent

    def gan_energy_loss(self, states, targets):
        '''Return the GAN energy loss. Put no variables here.'''
        ranges = []
        for _ in range(cfg.batch_size):
            ranges.append(tf.expand_dims(tf.range(tf.shape(states)[1]), 0))
        ranges = tf.concat(0, ranges)
        lengths = tf.expand_dims(self.lengths, -1)
        mask = tf.cast(tf.less(ranges, lengths), tf.float32)
        losses = tf.reduce_sum(tf.reduce_sum(tf.square(states - targets), [2]) * mask,
                               [1]) / tf.cast(lengths, tf.float32)
        if self.mle_mode:
            d_losses = losses
            g_losses = 0.0  # not useful in MLE mode
        else:
            # d_losses is max(0, m - loss) to prevent the manifold moving away from a far-away
            # generation, but g_losses still has to give a signal to the generator to move towards
            # the manifold.
            d_losses = tf.nn.relu(cfg.d_eb_margin - losses)
            g_losses = losses
        return d_losses, g_losses

    def gan_loss(self, d_out):
        '''Return the discriminator loss according to the label (1 if MLE mode).
           Put no variables here.'''
        return tf.nn.sigmoid_cross_entropy_with_logits(d_out, tf.constant(int(self.mle_mode),
                                                                          dtype=tf.float32,
                                                                          shape=d_out.get_shape()))

    def _train(self, cost, scope, optimizer, global_step=None):
        '''Generic training helper'''
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads = tf.gradients(cost, tvars)
        if cfg.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, cfg.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    def train_mle(self, cost):
        '''Training op for MLE mode.'''
        return self._train(cost, '.*/(Embeddings|Encoder|Decoder|MLE_Softmax)', self.g_optimizer,
                           self.global_step)

    def train_mle_encoder(self, cost):
        '''Encoder-only training op for MLE mode.'''
        return self._train(cost, '.*/(Embeddings|Encoder)', self.g_optimizer, self.global_step)

    def train_d(self, cost):
        '''Training op for GAN mode, discriminator.'''
        return self._train(cost, '.*/Discriminator', self.d_optimizer)

    def train_g(self, cost):
        '''Training op for GAN mode, generator.'''
        # don't update embeddings, just update the generated distributions
        return self._train(cost, '.*/(Transform_Latent|Decoder)', self.g_optimizer)
