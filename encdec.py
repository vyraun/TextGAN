import tensorflow as tf

import rnncell
import utils


class EncoderDecoderModel(object):
    '''The encoder-decoder adversarial model.'''

    def __init__(self, config, vocab, training, mle_mode, mle_reuse, gan_reuse):
        self.config = config
        self.vocab = vocab
        self.training = training
        self.mle_mode = mle_mode
        self.mle_reuse = mle_reuse
        self.gan_reuse = gan_reuse
        self.reuse = mle_reuse or gan_reuse # for common variables

        self.embedding = self.word_embedding_matrix()
        self.softmax_w, self.softmax_b = self.softmax_variables()

        if mle_mode:
            # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
            self.ldata = tf.placeholder(tf.int32, [config.batch_size, None], name='ldata')
            # right-aligned data: <pad...> <sos> w1 s2 ... w_T
            self.rdata = tf.placeholder(tf.int32, [config.batch_size, None], name='rdata')
            # sentence lengths
            self.lengths = tf.placeholder(tf.int32, [config.batch_size], name='lengths')
            # sentences with word dropout
            self.ldata_dropped = tf.placeholder(tf.int32, [config.batch_size, None],
                                                name='ldata_dropped')
            self.rdata_dropped = tf.placeholder(tf.int32, [config.batch_size, None],
                                                name='rdata_dropped')

            lembs = self.word_embeddings(self.ldata)
            rembs_dropped = self.word_embeddings(self.rdata_dropped)
            self.latent = self.encoder(rembs_dropped)
        else:
            # only the first timestep input will actually be considered
            lembs = self.word_embeddings(tf.constant(vocab.sos_index, shape=[config.batch_size, 1]))
            # so the rest can be zeros
            lembs = tf.concat(1, [lembs, tf.zeros([config.batch_size, config.gen_sent_length - 1,
                                                   config.word_emb_size])])
            self.rand_input = tf.placeholder(tf.float32, [config.batch_size, config.hidden_size],
                                             name='gan_random_input')
            self.latent = self.generate_latent(self.rand_input)
        output, states = self.decoder(lembs, self.latent)

        if not mle_mode:
            self.generated = tf.stop_gradient(self.output_words(output))
        d_out = self.discriminator(states)
        if mle_mode:
            # shift left the input to get the targets
            targets = tf.concat(1, [self.ldata[:,1:], tf.zeros([config.batch_size, 1], tf.int32)])
            mle_loss = self.mle_loss(output, targets)
            self.nll = tf.reduce_sum(mle_loss) / config.batch_size
            self.mle_cost = self.nll
            gan_loss = self.gan_loss(d_out, 1)
            self.gan_cost = tf.reduce_sum(gan_loss) / config.batch_size
            if training:
                self.train_op = [self.train_mle(self.mle_cost), tf.no_op()] # need to enable_gan
                self.gan_train_op = [self.train_op[0], self.train_d(self.gan_cost)]
            else:
                self.train_op = [tf.no_op(), tf.no_op()]
        else:
            gan_loss = self.gan_loss(d_out, 0)
            self.gan_cost = tf.reduce_sum(gan_loss) / config.batch_size
            self.train_op = [tf.no_op(), tf.no_op()] # need to explicitly enable_gan
            if training:
                self.gan_train_op = [self.train_g(-self.gan_cost), self.train_d(self.gan_cost)]

    def enable_gan(self):
        '''Enable GAN training.'''
        if self.training:
            self.train_op = self.gan_train_op

    def rnn_cell(self, latent=None, embedding=None, softmax_w=None, softmax_b=None,
                 return_states=False):
        '''Return a multi-layer RNN cell.'''
        softmax_top_k = self.config.generator_top_k
        if softmax_top_k > 0 and len(self.vocab.vocab) <= softmax_top_k:
            softmax_top_k = -1
        return rnncell.MultiRNNCell([rnncell.GRUCell(self.config.hidden_size, latent=latent)
                                         for _ in xrange(self.config.num_layers)],
                                    embedding=embedding, softmax_w=softmax_w, softmax_b=softmax_b,
                                    return_states=return_states, softmax_top_k=softmax_top_k)

    def word_embedding_matrix(self):
        '''Define the word embedding matrix.'''
        with tf.device('/cpu:0') and tf.variable_scope("Embeddings", reuse=self.reuse):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           self.config.word_emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
        return embedding

    def softmax_variables(self):
        '''Define the softmax weight and bias variables.'''
        with tf.variable_scope("MLE_Softmax", reuse=self.reuse):
            softmax_w = tf.get_variable("W", [len(self.vocab.vocab), self.config.hidden_size],
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
            rand_input = utils.highway(rand_input, layer_size=2)
            latent = utils.linear(rand_input, self.config.num_layers * self.config.hidden_size,
                                  True)
        return tf.nn.tanh(latent)

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation.'''
        with tf.variable_scope("Encoder", reuse=self.mle_reuse):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(), inputs, dtype=tf.float32)
            latent = utils.highway(state)
        return latent

    def decoder(self, inputs, latent):
        '''Use the latent representation and word inputs to predict next words.'''
        if self.config.force_nolatent:
            latent = None
        if self.config.force_noinputs:
            inputs = tf.zeros_like(inputs)
        with tf.variable_scope("Decoder", reuse=self.reuse):
            if self.mle_mode:
                outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(latent, return_states=True), inputs,
                                               dtype=tf.float32)
            else:
                outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(latent, self.embedding, self.softmax_w,
                                                             self.softmax_b, return_states=True),
                                               inputs, dtype=tf.float32)
            output = tf.slice(outputs, [0, 0, 0], [-1, -1, self.config.hidden_size])
            states = tf.slice(outputs, [0, 0, self.config.hidden_size], [-1, -1, -1])
            # for GRU, we skipped the last layer states because they're the outputs
            states = tf.concat(2, [states, output])
        return output, states

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        mask = tf.cast(tf.greater(targets, 0, name='targets_mask'), tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, self.config.hidden_size])
        if self.training and self.config.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, output, targets,
                                              self.config.softmax_samples, len(self.vocab.vocab))
            loss *= mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(self.softmax_w),
                                              name='softmax_transform_mle'), self.softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(targets, [-1])],
                                                          [tf.reshape(mask, [-1])])
        return tf.reshape(loss, [self.config.batch_size, -1])

    def output_words(self, outputs):
        '''Get output words from RNN outputs.'''
        output = tf.reshape(tf.concat(1, outputs), [-1, self.config.hidden_size])
        logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(self.softmax_w),
                                          name='softmax_transform_output'), self.softmax_b)
        logits = tf.reshape(logits, [self.config.batch_size, -1, len(self.vocab.vocab)])
        words = tf.slice(tf.cast(tf.argmax(logits, 2), tf.int32), [0, 0],
                         tf.pack([-1, tf.shape(outputs)[1] - 1]))
        return tf.concat(1, [words, tf.constant(self.vocab.eos_index,
                                                shape=[self.config.batch_size, 1])])

    def discriminator(self, states):
        '''Discriminator that operates on the final states of the sentences.'''
        with tf.variable_scope("Discriminator", reuse=self.reuse):
            if self.mle_mode:
                indices = self.lengths - 2
            else:
                eos_locs = tf.where(tf.equal(self.generated, self.vocab.eos_index))
                counts = tf.unique_with_counts(eos_locs[:,0])[2]
                meta_indices = tf.expand_dims(tf.cumsum(counts, exclusive=True), -1)
                gather_indices = tf.concat(1, [meta_indices, tf.ones_like(meta_indices)])
                indices = tf.cast(tf.gather_nd(eos_locs, gather_indices), tf.int32)
            final_states = utils.rowwise_lookup(states, indices) # 2D array of final states
            output = utils.linear(final_states, 1, True, 0.0, scope='discriminator_output')
        return output

    def gan_loss(self, d_out, label):
        '''Return the discriminator loss according to the label. Put no variables here.'''
        return tf.nn.sigmoid_cross_entropy_with_logits(d_out, tf.constant(label, dtype=tf.float32,
                                                                          shape=d_out.get_shape()))

    def _train(self, lr, cost, scope):
        '''Generic training helper'''
        optimizer = utils.get_optimizer(self.config, lr)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads = tf.gradients(cost, tvars)
        if self.config.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars))

    def train_mle(self, cost):
        '''Training op for MLE mode.'''
        with tf.variable_scope("LR", reuse=self.mle_reuse):
            self.mle_lr = tf.get_variable("mle_lr", shape=[], initializer=tf.zeros_initializer,
                                          trainable=False)
        return self._train(self.mle_lr, cost, '.*/(Embeddings|Encoder|Decoder|MLE_Softmax)')

    def train_d(self, cost):
        '''Training op for GAN mode, discriminator.'''
        with tf.variable_scope("LR", reuse=self.reuse):
            self.d_lr = tf.get_variable("d_lr", shape=[], initializer=tf.zeros_initializer,
                                        trainable=False)
        return self._train(self.d_lr, cost, '.*/Discriminator')

    def train_g(self, cost):
        '''Training op for GAN mode, generator.'''
        with tf.variable_scope("LR", reuse=self.gan_reuse):
            self.g_lr = tf.get_variable("g_lr", shape=[], initializer=tf.zeros_initializer,
                                        trainable=False)
        # don't update embeddings, just update the generated distributions
        return self._train(self.g_lr, cost, '.*/(Transform_Latent|Decoder)')

    def assign_mle_lr(self, session, lr_value):
        '''Change the MLE learning rate'''
        print 'Setting MLE learning rate to', lr_value
        session.run(tf.assign(self.mle_lr, lr_value))

    def assign_d_lr(self, session, lr_value):
        '''Change the discriminator learning rate'''
        print 'Setting discriminator learning rate to', lr_value
        session.run(tf.assign(self.d_lr, lr_value))

    def assign_g_lr(self, session, lr_value):
        '''Change the generator learning rate'''
        print 'Setting generator learning rate to', lr_value
        session.run(tf.assign(self.g_lr, lr_value))
