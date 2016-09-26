from __future__ import division

import numpy as np
import tensorflow as tf

from config import Config
import reader


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


def call_session(session, model, batch):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.ldata: batch[0], model.rdata: batch[1]}
    ret = session.run([m.perplexity, m.cost, m.train_op], f_dict)
    return ret[:-1]


def run_epoch(session, model, config, vocab, saver, steps):
    '''Runs the model on the given data for an epoch.'''
    rd = reader.Reader(config, vocab)
    if config.training:
        batch_loader = rd.training()
    else:
        batch_loader = rd.validation()
    start_time = time.time()
    perps = 0.0
    costs = 0.0
    iters = 0
    shortterm_perps = 0.0
    shortterm_costs = 0.0
    shortterm_iters = 0

    for step, batch in enumerate(batch_loader):
        perp, cost = call_session(session, model, batch)

        perps += perp
        costs += cost
        shortterm_perps += perp
        shortterm_costs += cost
        iters += batch.shape[1]
        shortterm_iters += batch.shape[1]

        if step % config.print_every == 0:
            avg_perp = shortterm_perps / shortterm_iters
            avg_cost = shortterm_costs / shortterm_iters
            print("%d  perplexity: %.3f  ml_loss: %.4f  cost: %.4f  speed: %.0f wps" %
                  (step, np.exp(avg_perp), avg_perp, avg_cost,
                   shortterm_iters * config.batch_size / (time.time() - start_time)))

            shortterm_perps = 0.0
            shortterm_costs = 0.0
            shortterm_iters = 0
            start_time = time.time()

        cur_iters = steps + step
        if config.training and cur_iters and cur_iters % config.save_every == 0:
            save_file = config.save_file
            if not config.save_overwrite:
                save_file = save_file + '.' + str(cur_iters)
            print "Saving model (epoch perplexity: %.3f) ..." % np.exp(perps / iters)
            save_file = saver.save(session, save_file)
            print "Saved to", save_file

        if cur_iters >= config.max_steps:
            break

    return np.exp(perps / iters), steps + step


def main(_):
    config = Config()
    vocab = reader.Vocab(config)
    vocab.load_from_pickle()

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        with tf.variable_scope("model", reuse=None):
            model = EncoderDecoderModel(config, vocab)
        saver = tf.train.Saver()
        try:
            saver.restore(session, config.load_file)
            print "Model restored from", config.load_file
        except ValueError:
            if config.training:
                tf.initialize_all_variables().run()
                print "No loadable model file, new model initialized."
            else:
                print "You need to provide a valid model file for testing!"
                sys.exit(1)

        steps = 0
        if config.training:
            model.assign_lr(session, config.learning_rate)
        for i in xrange(config.max_epoch):
            if config.training:
                print "Epoch: %d Learning rate: %.4f" % (i + 1, session.run(model.lr))
            perplexity, steps = run_epoch(session, model, config, vocab, saver, steps)
            if config.training:
                print "Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity)
            else:
                print "Test Perplexity: %.3f" % (perplexity,)
                break
            if steps >= config.max_steps:
                break


if __name__ == "__main__":
    tf.app.run()
