import tensorflow as tf

from config import cfg
import rnncell
import utils


class EncoderDecoderModel(object):

    '''The variational encoder-decoder adversarial model.'''

    def __init__(self, vocab, training, use_gan=True, g_optimizer=None, d_optimizer=None,
                 generator=False):
        self.vocab = vocab
        self.training = training
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.embedding = self.word_embedding_matrix()
        self.softmax_w, self.softmax_b = self.softmax_variables()

        with tf.variable_scope("GlobalMLE"):
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.zeros_initializer,
                                               trainable=False)
        # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
        self.data = tf.placeholder(tf.int32, [cfg.batch_size, None], name='data')
        # sentence lengths
        self.lengths = tf.placeholder(tf.int32, [cfg.batch_size], name='lengths')
        # sentences with word dropout
        self.data_dropped = tf.placeholder(tf.int32, [cfg.batch_size, None], name='data_dropped')

        embs = self.word_embeddings(self.data)
        embs_dropped = self.word_embeddings(self.data_dropped)
        embs_reversed = tf.reverse_sequence(embs, self.lengths, 1)
        z_mean, z_logvar = self.encoder(embs_reversed)
        if cfg.variational:
            eps = tf.random_normal([cfg.batch_size, cfg.latent_size])
            self.latent = z_mean + tf.mul(tf.sqrt(tf.exp(z_logvar)), eps)
        else:
            self.latent = z_mean

        if generator:
            self.latent = tf.placeholder(tf.float32, [cfg.batch_size, cfg.latent_size],
                                         name='generator_input')

        output, mle_states, _ = self.decoder(embs_dropped, True)
        _, gan_states, self.generated = self.decoder(embs_dropped, False, True)
        if use_gan:
            states = tf.concat(0, [mle_states, gan_states])

            if cfg.d_energy_based:
                d_out, d_latent = self.discriminator_energy(states)
                # shift left the states to get the targets
                targets = tf.concat(1, [states, tf.expand_dims(d_latent, 1)])
                d_loss, g_loss = self.gan_energy_loss(d_out, targets)
                self.d_cost = tf.reduce_sum(d_loss) / (2 * cfg.batch_size)
                self.g_cost = tf.reduce_sum(g_loss) / (2 * cfg.batch_size)
            else:
                if cfg.d_rnn:
                    d_out = self.discriminator_rnn(states)
                else:
                    d_out = self.discriminator_finalstate(states)
                targets = tf.concat(0, [tf.ones([cfg.batch_size, 1]),
                                        tf.zeros([cfg.batch_size, 1])])
                gan_loss = self.gan_loss(d_out, targets)
                self.d_cost = tf.reduce_sum(gan_loss) / (2 * cfg.batch_size)
                self.g_cost = -self.d_cost
        else:
            self.d_cost = tf.zeros([])
            self.g_cost = tf.zeros([])

        # shift left the input to get the targets
        targets = tf.concat(1, [self.data[:, 1:], tf.zeros([cfg.batch_size, 1], tf.int32)])
        self.nll = tf.reduce_sum(self.mle_loss(output, targets)) / cfg.batch_size
        if cfg.variational:
            self.kld = tf.reduce_sum(self.kld_loss(z_mean, z_logvar)) / cfg.batch_size
        else:
            self.kld = tf.zeros([])
        self.kld_weight = cfg.anneal_max * tf.sigmoid((7 / cfg.anneal_bias) *
                                                      (self.global_step - cfg.anneal_bias))
        self.mle_cost = self.nll + (self.kld_weight * self.kld)
        if training:
            self.mle_train_op = self.train_mle(self.mle_cost)
        else:
            self.mle_train_op = tf.no_op()
        if training and use_gan:
            self.d_train_op = self.train_d(self.d_cost)
            self.g_train_op = self.train_g(self.g_cost)
        else:
            self.d_train_op = tf.no_op()
            self.g_train_op = tf.no_op()

    def rnn_cell(self, num_layers, hidden_size, latent=None, embedding=None, softmax_w=None,
                 softmax_b=None, return_states=False, pretanh=False, get_embeddings=False):
        '''Return a multi-layer RNN cell.'''
        softmax_top_k = cfg.generator_top_k
        if softmax_top_k > 0 and len(self.vocab.vocab) <= softmax_top_k:
            softmax_top_k = -1
        return rnncell.MultiRNNCell([rnncell.GRUCell(hidden_size, pretanh=pretanh)
                                     for _ in range(num_layers)], latent=latent,
                                    embedding=embedding, softmax_w=softmax_w, softmax_b=softmax_b,
                                    return_states=return_states, softmax_top_k=softmax_top_k,
                                    use_argmax=cfg.generate_argmax, pretanh=pretanh,
                                    get_embeddings=get_embeddings)

    def word_embedding_matrix(self):
        '''Define the word embedding matrix.'''
        with tf.device('/cpu:0') and tf.variable_scope("Embeddings"):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           cfg.emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
        return embedding

    def softmax_variables(self):
        '''Define the softmax weight and bias variables.'''
        with tf.variable_scope("MLE_Softmax"):
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
        with tf.variable_scope("Encoder"):
            if cfg.enc_bidirect:
                fcell = self.rnn_cell(cfg.num_layers, cfg.hidden_size, return_states=True)
                bcell = self.rnn_cell(cfg.num_layers, cfg.hidden_size, return_states=True)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(fcell, bcell, inputs,
                                                             sequence_length=self.lengths,
                                                             swap_memory=True, dtype=tf.float32)
            else:
                cell = self.rnn_cell(cfg.num_layers, cfg.hidden_size, return_states=True)
                outputs, _ = tf.nn.dynamic_rnn(cell, inputs, swap_memory=True, dtype=tf.float32)
                outputs = (outputs,)  # to match bidirectional RNN's output format
            states = []
            for out in outputs:
                output = out[:, :, :cfg.hidden_size]
                d_states = out[:, :, cfg.hidden_size:]
                # for GRU, we skipped the last layer states because they're the outputs
                states.append(tf.concat(2, [d_states, output]))
            states = tf.concat(2, states)  # concatenated states from fwd and bwd RNNs
            states = tf.reshape(states, [-1, cfg.hidden_size * len(outputs)])
            states = utils.linear(states, cfg.latent_size, True, 0.0, scope='states_transform')
            states = utils.highway(states, f=tf.nn.elu)
            states = tf.reshape(states, [cfg.batch_size, -1, cfg.latent_size])
            latent = tf.reduce_sum(states, [1])
            # XXX these linears are prone to exploding the KL divergence
            z_mean = utils.linear(latent, cfg.latent_size, True, 0.0, scope='Latent_mean')
            z_logvar = utils.linear(latent, cfg.latent_size, True, 0.0, scope='Latent_logvar')
        return z_mean, z_logvar

    def decoder(self, inputs, mle_mode, reuse=None):
        '''Use the latent representation and word inputs to predict next words.'''
        with tf.variable_scope("Decoder", reuse=reuse):
            latent = utils.highway(self.latent, layer_size=2, f=tf.nn.elu)
            latent = utils.linear(latent, cfg.latent_size, True, 0.0, scope='Latent_transform')
            initial = []
            for i in range(cfg.num_layers):
                preact = utils.linear(latent, cfg.hidden_size, True, 0.0,
                                      scope='Latent_initial%d' % i)
                act = tf.nn.tanh(preact)
                initial.append(tf.concat(1, [act, preact]))
            if mle_mode:
                inputs = tf.concat(2, [inputs, tf.tile(tf.expand_dims(latent, 1),
                                                       tf.pack([1, tf.shape(inputs)[1], 1]))])
                cell = self.rnn_cell(cfg.num_layers, cfg.hidden_size, return_states=True,
                                     pretanh=True)
            else:
                cell = self.rnn_cell(cfg.num_layers, cfg.hidden_size, latent, self.embedding,
                                     self.softmax_w, self.softmax_b, return_states=True,
                                     pretanh=True, get_embeddings=cfg.concat_inputs)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, initial_state=cell.initial_state(initial),
                                           swap_memory=True, dtype=tf.float32)
            output = outputs[:, :, :cfg.hidden_size]
            if mle_mode:
                generated = None
                skip = 0
            else:
                words = tf.squeeze(tf.cast(outputs[:, :-1, cfg.hidden_size:cfg.hidden_size+1],
                                           tf.int32), [-1])
                generated = tf.stop_gradient(tf.concat(1, [words, tf.constant(self.vocab.eos_index,
                                                                       shape=[cfg.batch_size, 1])]))
                skip = 1
                if cfg.concat_inputs:
                    embeddings = outputs[:, :, cfg.hidden_size+1:cfg.hidden_size+1+cfg.emb_size]
                    embeddings = tf.concat(1, [inputs[:, :1, :], embeddings[:, :-1, :]])
                    embeddings = tf.concat(2, [embeddings, tf.tile(tf.expand_dims(latent, 1),
                                                                   tf.pack([1,
                                                                            tf.shape(embeddings)[1],
                                                                            1]))])
                    skip += cfg.emb_size
            states = outputs[:, :, cfg.hidden_size+skip:]
            if cfg.concat_inputs:
                if mle_mode:
                    states = tf.concat(2, [states, inputs])
                else:
                    states = tf.concat(2, [states, embeddings])
        return output, states, generated

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        present_mask = tf.greater(targets, 0, name='present_mask')
        # don't enfoce loss on true <unk>'s
        unk_mask = tf.not_equal(targets, self.vocab.unk_index, name='unk_mask')
        mask = tf.cast(tf.logical_and(present_mask, unk_mask), tf.float32)
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

    def discriminator_rnn(self, states):
        '''Recurrent discriminator that operates on the sequence of states of the sentences.'''
        with tf.variable_scope("Discriminator"):
            if cfg.d_rnn_bidirect:
                hidden_size = cfg.hidden_size
                fcell = self.rnn_cell(cfg.d_num_layers, hidden_size, return_states=True)
                bcell = self.rnn_cell(cfg.d_num_layers, hidden_size, return_states=True)
                seq_lengths = tf.pack([tf.shape(states)[1]] * (2 * cfg.batch_size))
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(fcell, bcell, states,
                                                             sequence_length=seq_lengths,
                                                             swap_memory=True, dtype=tf.float32)
            else:
                hidden_size = cfg.hidden_size * 2
                cell = self.rnn_cell(cfg.d_num_layers, hidden_size, return_states=True)
                outputs, _ = tf.nn.dynamic_rnn(cell, states, swap_memory=True, dtype=tf.float32)
                outputs = (outputs,)  # to match bidirectional RNN's output format
            d_states = []
            for out in outputs:
                output = out[:, :, :hidden_size]
                dir_states = out[:, :, hidden_size:]
                # for GRU, we skipped the last layer states because they're the outputs
                d_states.append(tf.concat(2, [dir_states, output]))
        return self._discriminator_conv(tf.concat(2, d_states))

    def _discriminator_conv(self, states):
        '''Convolve output of bidirectional RNN and predict the discriminator label.'''
        with tf.variable_scope("Discriminator"):
            W_conv = tf.get_variable('W_conv', [cfg.d_conv_window, 1, states.get_shape()[2],
                                                cfg.hidden_size // cfg.d_conv_window],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_conv = tf.get_variable('b_conv', [cfg.hidden_size // cfg.d_conv_window],
                                     initializer=tf.constant_initializer(0.0))
            states = tf.expand_dims(states, 2)
            conv = tf.nn.conv2d(states, W_conv, strides=[1, 1, 1, 1], padding='SAME')
            conv_out = tf.reshape(conv, [2 * cfg.batch_size, -1,
                                         cfg.hidden_size // cfg.d_conv_window])
            conv_out = tf.nn.elu(tf.nn.bias_add(conv_out, b_conv))
            reduced = tf.reduce_sum(conv_out, [1])
            output = utils.linear(reduced, 1, True, 0.0, scope='discriminator_output')
        return output

    def discriminator_finalstate(self, states):  # FIXME
        '''Discriminator that operates on the final states of the sentences.'''
        with tf.variable_scope("Discriminator"):
            # indices = lengths - 2, since the generated output skips <sos>
            #final_states = utils.rowwise_lookup(states, self.lengths - 2)
            final_states = states[:, -1, :]
            combined = tf.concat(1, [self.latent, final_states])  # TODO transform latent
            lin1 = tf.nn.elu(utils.linear(combined, cfg.hidden_size, True, 0.0,
                                          scope='discriminator_lin1'))
            lin2 = tf.nn.elu(utils.linear(lin1, cfg.hidden_size // 2, True, 0.0,
                                          scope='discriminator_lin2'))
            output = utils.linear(lin2, 1, True, 0.0, scope='discriminator_output')
        return output

    def discriminator_energy(self, states):  # FIXME
        '''An energy-based discriminator that tries to reconstruct the input states.'''
        with tf.variable_scope("Discriminator"):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(cfg.d_num_layers, cfg.hidden_size), states,
                                         swap_memory=True, dtype=tf.float32,
                                         scope='discriminator_encoder')
            # XXX use BiRNN+convnet for the encoder
            # this latent is of size cfg.hidden_size since it needs a lot more capacity than
            # cfg.latent_size to reproduce the hidden states
            # TODO use all states instead of just the final state
            latent = utils.highway(state, layer_size=1)
            latent = utils.linear(latent, cfg.hidden_size, True,
                                  scope='discriminator_latent_transform')
            # TODO make initial state from latent, don't just use zeros
            decoder_input = tf.concat(1, [tf.zeros([2 * cfg.batch_size, 1, cfg.hidden_size]),
                                      states])
            output, _ = tf.nn.dynamic_rnn(self.rnn_cell(cfg.d_num_layers, cfg.hidden_size, latent),
                                          decoder_input, swap_memory=True, dtype=tf.float32,
                                          scope='discriminator_decoder')
            output = tf.reshape(output, [-1, cfg.hidden_size])
            reconstructed = utils.linear(output, cfg.hidden_size, True, 0.0,
                                         scope='discriminator_reconst')
            reconstructed = tf.reshape(reconstructed, [2 * cfg.batch_size, -1, cfg.hidden_size])
            # don't train this projection, since the model can learn to zero out ret_latent to
            # minimize the reconstruction error
            ret_latent = tf.nn.tanh(utils.linear(self.latent, cfg.hidden_size, False,
                                                 scope='discriminator_ret_latent', train=False))
        return reconstructed, ret_latent

    def gan_energy_loss(self, states, targets):  # FIXME
        '''Return the GAN energy loss. Put no variables here.'''
        # TODO fix every occurence of max_sent_length
        losses = tf.reduce_sum(tf.square(states - targets), [1, 2]) / cfg.max_sent_length
        d_losses = losses[:cfg.batch_size] + tf.nn.relu(cfg.d_eb_margin - losses[cfg.batch_size:])
        g_losses = losses[cfg.batch_size:]
        return d_losses, g_losses

    def gan_loss(self, d_out, targets):
        '''Return the discriminator loss according to the label (1 if MLE mode).
           Put no variables here.'''
        return tf.nn.sigmoid_cross_entropy_with_logits(d_out, targets)

    def _train(self, cost, scope, optimizer, global_step=None):
        '''Generic training helper'''
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads = tf.gradients(cost, tvars)
        if cfg.max_grad_norm > 0:
            # TODO do clip by global norm for all vars before subsetting
            grads, _ = tf.clip_by_global_norm(grads, cfg.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    def train_mle(self, cost):
        '''Training op for MLE mode.'''
        return self._train(cost, '.*/(Embeddings|Encoder|Decoder|MLE_Softmax)', self.g_optimizer,
                           self.global_step)

    def train_d(self, cost):
        '''Training op for GAN mode, discriminator.'''
        return self._train(cost, '.*/Discriminator', self.d_optimizer)

    def train_g(self, cost):
        '''Training op for GAN mode, generator.'''
        # don't update embeddings, just update the generated distributions
        return self._train(cost, '.*/Decoder', self.g_optimizer)
