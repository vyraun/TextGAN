import tensorflow as tf

import rnncell
import utils


class EncoderDecoderModel(object):
    '''The encoder-decoder adversarial model.'''

    def __init__(self, config, vocab, training, mle_mode):
        self.config = config
        self.vocab = vocab
        self.training = training
        self.mle_mode = mle_mode

        if mle_mode:
            # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
            self.ldata = tf.placeholder(tf.int32, [config.batch_size, None], name='ldata')
            # right-aligned data: <pad...> <sos> w1 s2 ... w_T
            self.rdata = tf.placeholder(tf.int32, [config.batch_size, None], name='rdata')
            # sentence lengths
            self.lengths = tf.placeholder(tf.int32, [config.batch_size], name='lengths')

            lembs_dropped = self.word_embeddings(self.word_dropout(self.ldata))
            rembs = self.word_embeddings(self.rdata, reuse=True)
            self.latent = self.encoder(rembs)
        else:
            # TODO this is incorrect! can't give proper inputs at MLE and zeros for GAN.
            lembs_dropped = tf.zeros([config.batch_size, config.gen_sent_length,
                                      config.word_emb_size])
            self.latent = tf.placeholder(tf.float32, [config.batch_size,
                                                      config.num_layers * config.hidden_size],
                                         name='gan_random_input')
        outputs = self.decoder(lembs_dropped, self.latent)
        d_out = self.discriminator(outputs)
        if mle_mode:
            # shift left the input to get the targets
            targets = tf.concat(1, [self.ldata[:,1:], tf.zeros([config.batch_size, 1], tf.int32)])
            mle_loss = self.mle_loss(outputs, targets)
            self.nll = tf.reduce_sum(mle_loss) / config.batch_size
            self.mle_cost = self.nll
            gan_loss = self.gan_loss(d_out, 1)
            self.gan_cost = tf.reduce_sum(gan_loss) / config.batch_size
            if training:
                self.train_op = [self.train_mle(self.mle_cost), self.train_d(self.gan_cost)]
            else:
                self.train_op = [tf.no_op(), tf.no_op()]
        else:
            gan_loss = self.gan_loss(d_out, 0)
            self.gan_cost = tf.reduce_sum(gan_loss) / config.batch_size
            self.train_op = [self.train_g(self.gan_cost), self.train_d(self.gan_cost)]

    def rnn_cell(self, latent=None):
        '''Return a multi-layer RNN cell.'''
        return tf.nn.rnn_cell.MultiRNNCell([rnncell.GRUCell(self.config.hidden_size, latent=latent)
                                               for _ in xrange(self.config.num_layers)],
                                           state_is_tuple=True)

    def word_dropout(self, inputs):
        '''Randomly replace words from inputs with <unk>.'''
        if self.training and self.config.decoder_dropout > 0.0:
            unks = tf.ones_like(inputs, tf.int32) * self.vocab.unk_index
            mask = tf.cast(tf.greater(tf.nn.dropout(tf.cast(unks, tf.float32),
                                                    self.config.decoder_dropout), 0.0), tf.int32)
            return ((1 - mask) * inputs) + (mask * unks)
        else:
            return inputs

    def word_embeddings(self, inputs, reuse=False):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0') and tf.variable_scope("Embeddings", reuse=reuse):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           self.config.word_emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embeds = tf.nn.embedding_lookup(embedding, inputs, name='word_embedding_lookup')
        return embeds

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation.'''
        with tf.variable_scope("Encoder"):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(), inputs, dtype=tf.float32)
            latent = utils.highway(state)
        return latent

    def decoder(self, inputs, latent):
        '''Use the latent representation and word inputs to predict next words.'''
        if self.config.force_nolatent:
            latent = None
        if self.config.force_noinputs:
            inputs = tf.zeros_like(inputs)
        with tf.variable_scope("Decoder"):
            outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(latent), inputs, dtype=tf.float32)
        return outputs

    def _mle_softmax_vars(self, reuse=True):
        with tf.variable_scope("MLE_Softmax", reuse=reuse):
            softmax_w = tf.get_variable("W", [len(self.vocab.vocab), self.config.hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("b", [len(self.vocab.vocab)],
                                        initializer=tf.zeros_initializer)
        return softmax_w, softmax_b

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        mask = tf.cast(tf.greater(targets, 0, name='targets_mask'), tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, self.config.hidden_size])
        softmax_w, softmax_b = self._mle_softmax_vars(None)
        if self.training and self.config.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, output, targets,
                                              self.config.softmax_samples, len(self.vocab.vocab))
            loss *= mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(softmax_w),
                                              name='softmax_transform_mle'), softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(targets, [-1])],
                                                          [tf.reshape(mask, [-1])])
        return tf.reshape(loss, [self.config.batch_size, -1])

    def output_words(self, outputs):
        softmax_w, softmax_b = self._mle_softmax_vars()
        logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(softmax_w),
                                          name='softmax_transform_output'), softmax_b)

    def discriminator(self, states):
        '''Discriminator that operates on the final states of the sentences.'''
        with tf.variable_scope("Discriminator"):
            if self.mle_mode:
                indices = self.lengths - 2
            else:
                # TODO get sentence lengths using MLE softmax
                indices = tf.constant(self.config.gen_sent_length - 1,
                                      shape=[self.config.batch_size])
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
        self.mle_lr = tf.Variable(0.0, trainable=False)
        return self._train(self.mle_lr, cost, '.*/(Embeddings|Encoder|Decoder|MLE_Softmax)')

    def train_d(self, cost):
        '''Training op for GAN mode, discriminator.'''
        self.d_lr = tf.Variable(0.0, trainable=False)
        return self._train(self.d_lr, cost, '.*/Discriminator')

    def train_g(self, cost):
        '''Training op for GAN mode, generator.'''
        self.g_lr = tf.Variable(0.0, trainable=False)
        return self._train(self.g_lr, cost, '.*/Decoder') # XXX embeddings?

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
