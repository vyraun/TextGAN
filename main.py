from __future__ import division

import time

import numpy as np
import tensorflow as tf

from config import Config
from encdec import EncoderDecoderModel
from reader import Reader, Vocab


def call_session(session, model, batch):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.ldata: batch[0], model.rdata: batch[1]}
    ret = session.run([model.nll, model.cost, model.train_op], f_dict)
    return ret[:-1]


def run_epoch(session, model, config, vocab, saver, steps):
    '''Runs the model on the given data for an epoch.'''
    reader = Reader(config, vocab)
    if config.training:
        batch_loader = reader.training()
    else:
        batch_loader = reader.validation()
    start_time = time.time()
    nlls = 0.0
    costs = 0.0
    iters = 0
    shortterm_nlls = 0.0
    shortterm_costs = 0.0
    shortterm_iters = 0

    for step, batch in enumerate(batch_loader):
        nll, cost = call_session(session, model, batch)

        nlls += nll
        costs += cost
        shortterm_nlls += nll
        shortterm_costs += cost
        iters += batch.shape[1]
        shortterm_iters += batch.shape[1]

        if step % config.print_every == 0:
            avg_nll = shortterm_nlls / shortterm_iters
            avg_cost = shortterm_costs / shortterm_iters
            print("%d  perplexity: %.3f  ml_loss: %.4f  cost: %.4f  speed: %.0f wps" %
                  (step, np.exp(avg_nll), avg_nll, avg_cost,
                   shortterm_iters * config.batch_size / (time.time() - start_time)))

            shortterm_nlls = 0.0
            shortterm_costs = 0.0
            shortterm_iters = 0
            start_time = time.time()

        cur_iters = steps + step
        if config.training and cur_iters and cur_iters % config.save_every == 0:
            save_file = config.save_file
            if not config.save_overwrite:
                save_file = save_file + '.' + str(cur_iters)
            print "Saving model (epoch perplexity: %.3f) ..." % np.exp(nlls / iters)
            save_file = saver.save(session, save_file)
            print "Saved to", save_file

        if cur_iters >= config.max_steps:
            break

    return np.exp(nlls / iters), steps + step


def main(_):
    config = Config()
    vocab = Vocab(config)
    vocab.load_from_pickle()

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        with tf.variable_scope("Model", reuse=None):
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
