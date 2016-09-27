import tensorflow as tf


class EncoderDecoderModel(object):
    '''The encoder-decoder model.'''

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
        ldata = tf.placeholder(tf.int32, [config.batch_size, None], name='ldata')
        # right-aligned data: <pad...> <sos> w1 s2 ... w_T <eos>
        rdata = tf.placeholder(tf.int32, [config.batch_size, None], name='rdata')
        # masks where padding words are 0 and all others are 1
        ldata_mask = tf.greater(ldata, 0, name='ldata_mask')
        rdata_mask = tf.greater(rdata, 0, name='rdata_mask')

        sent_length = tf.shape(ldata)[1]
        lembs = self.word_embeddings(ldata)
        rembs = self.word_embeddings(rdata)
        latent = self.encoder(tf.slice(rembs, [0, 0, 0], tf.pack([-1, sent_length-1, -1])))
        outputs = self.decoder(lembs, latent)
        loss = self.mle_loss(outputs, tf.slice(ldata, [0, 1], tf.pack([-1, sent_length-1])))
        self.cost = loss / config.batch_size
        if config.training:
            self.train_op = self.train(self.cost)
        else:
            self.train_op = tf.no_op()

    def rnn_cell(self):
        '''Return a multi-layer RNN cell.'''
        gru_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        return tf.nn.rnn_cell.MultiRNNCell([gru_cell] * self.config.num_layers)

    def word_embeddings(self, inputs):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           self.config.word_emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embeds = tf.nn.embedding_lookup(embedding, inputs, name='word_embedding_lookup')
        return embeds

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation.'''
        pass # TODO

    def decoder(self, inputs, latent):
        '''Use the latent representation and word inputs to predict next words.'''
        # TODO use word-dropout for inputs so that the model learns to use $latent.
        pass # TODO

    def mle_loss(self, outputs, truth):
        '''Maximum likelihood estimation loss.'''
        pass # TODO

    def train(self, cost):
        '''Training op.'''
        self.lr = tf.Variable(0.0, trainable=False)
        if config.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        if self.config.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        '''Change the learning rate'''
        print 'Setting learning rate to', lr_value
        session.run(tf.assign(self.lr, lr_value))
