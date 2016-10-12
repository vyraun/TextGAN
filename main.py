from __future__ import division

import sys
import time

import numpy as np
import tensorflow as tf

from config import Config
from encdec import EncoderDecoderModel
from reader import Reader, Vocab


def call_mle_session(session, model, batch):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.ldata: batch[0],
              model.rdata: batch[1],
              model.ldata_dropped: batch[2],
              model.rdata_dropped: batch[3],
              model.lengths: batch[4]}
    # model.train_op will be tf.no_op() for a non-training model
    return session.run([model.nll, model.mle_cost, model.gan_cost, model.train_op], f_dict)[:-1]


def get_latent_representation(latent_dims):
    '''Generate a random latent representation to generate text from.'''
    return np.clip(np.random.normal(scale=0.5, size=latent_dims), -1.0, 1.0)


def call_gan_session(session, model, latent_dims):
    '''Use the session to train the generator of the GAN with fake samples.'''
    # XXX z from normal distribution may not be a good assumption, since encoded samples may not
    #     come from that. encourage encoder outputs to live in this prior?
    f_dict = {model.latent: get_latent_representation(latent_dims)}
    # model.train_op will be tf.no_op() for a non-training model
    return session.run([model.gan_cost, model.train_op], f_dict)[:-1]


def generate_sentences(session, model, latent_dims, vocab):
    '''Generate novel sentences using the generator.'''
    f_dict = {model.latent: get_latent_representation(latent_dims)}
    output = session.run(model.generated, f_dict)
    print '\nVisualizing new batch!'
    for i, sent in enumerate(output):
        print 'Sentence %d:' % i,
        for word in sent:
            if word == vocab.eos_index:
                break
            print vocab.vocab[word],
        print
    print


def save_model(session, saver, config, perp, cur_iters):
    '''Save model file.'''
    save_file = config.save_file
    if not config.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print "Saving model (epoch perplexity: %.3f) ..." % perp
    save_file = saver.save(session, save_file)
    print "Saved to", save_file


def run_epoch(session, mle_model, gan_model, batch_loader, config, vocab, saver, steps, max_steps,
              gen_samples=0):
    '''Runs the model on the given data for an epoch.'''
    start_time = time.time()
    nlls = 0.0
    mle_costs = 0.0
    gan_costs = 0.0
    iters = 0
    shortterm_nlls = 0.0
    shortterm_mle_costs = 0.0
    shortterm_gan_costs = 0.0
    shortterm_iters = 0

    for step, batch in enumerate(batch_loader):
        nll, mle_cost, d_cost = call_mle_session(session, mle_model, batch)
        g_cost = call_gan_session(session, gan_model, [config.batch_size,
                                                       config.num_layers * config.hidden_size])
        gan_cost = (g_cost + d_cost) / 2

        nlls += nll
        mle_costs += mle_cost
        gan_costs += gan_cost
        shortterm_nlls += nll
        shortterm_mle_costs += mle_cost
        shortterm_gan_costs += gan_cost
        # batch[1] is the right aligned batch, without <eos>. predictions also have one token less.
        iters += batch[1].shape[1]
        shortterm_iters += batch[1].shape[1]

        if step % config.print_every == 0:
            avg_nll = shortterm_nlls / shortterm_iters
            avg_mle_cost = shortterm_mle_costs / shortterm_iters
            avg_gan_cost = shortterm_gan_costs / shortterm_iters
            print("%d  perplexity: %.3f  mle_loss: %.4f  mle_cost: %.4f  gan_cost: %.4f  "
                  "speed: %.0f wps" %
                  (step, np.exp(avg_nll), avg_nll, avg_mle_cost, avg_gan_cost,
                   shortterm_iters * config.batch_size / (time.time() - start_time)))

            shortterm_nlls = 0.0
            shortterm_mle_costs = 0.0
            shortterm_gan_costs = 0.0
            shortterm_iters = 0
            start_time = time.time()

        cur_iters = steps + step
        if saver is not None and cur_iters and config.save_every > 0 and \
                cur_iters % config.save_every == 0:
            save_model(session, saver, config, np.exp(nlls / iters), cur_iters)

        if max_steps > 0 and cur_iters >= max_steps:
            break

    for _ in xrange(gen_samples):
        generate_sentences(session, gan_model, [config.batch_size,
                                                config.num_layers * config.hidden_size], vocab)
    perp = np.exp(nlls / iters)
    cur_iters = steps + step
    if saver is not None and config.save_every < 0:
        save_model(session, saver, config, perp, cur_iters)
    return perp, cur_iters


def main(_):
    config = Config()
    vocab = Vocab(config)
    vocab.load_from_pickle()
    reader = Reader(config, vocab)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        if config.training:
            with tf.variable_scope("Model", reuse=None):
                mle_model = EncoderDecoderModel(config, vocab, True, True)
            with tf.variable_scope("Model", reuse=True):
                gan_model = EncoderDecoderModel(config, vocab, True, False)
                eval_mle_model = EncoderDecoderModel(config, vocab, False, True)
                eval_gan_model = EncoderDecoderModel(config, vocab, False, False)
        else:
            with tf.variable_scope("Model", reuse=None):
                test_mle_model = EncoderDecoderModel(config, vocab, False, True)
            with tf.variable_scope("Model", reuse=True):
                test_gan_model = EncoderDecoderModel(config, vocab, False, False)
        saver = tf.train.Saver()
        try:
            # try to restore a saved model file
            saver.restore(session, config.load_file)
            print "Model restored from", config.load_file
        except ValueError:
            if config.training:
                tf.initialize_all_variables().run()
                print "No loadable model file, new model initialized."
            else:
                print "You need to provide a valid model file for testing!"
                sys.exit(1)

        if config.training:
            steps = 0
            train_perps = []
            valid_perps = []
            mle_model.assign_mle_lr(session, config.mle_learning_rate)
            gan_model.assign_d_lr(session, config.d_learning_rate)
            gan_model.assign_g_lr(session, config.g_learning_rate)
            mle_model.enable_gan() # TODO do this after config.gan_wait_epochs epochs
            gan_model.enable_gan()
            for i in xrange(config.max_epoch):
                print "\nEpoch: %d MLE learning rate: %.4f, D learning rate: %.4f, " \
                      "G learning rate: %.4f" % (i + 1, session.run(mle_model.mle_lr),
                                           session.run(gan_model.d_lr), session.run(gan_model.g_lr))
                perplexity, steps = run_epoch(session, mle_model, gan_model, reader.training(),
                                              config, vocab, saver, steps, config.max_steps)
                print "Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity)
                train_perps.append(perplexity)
                if config.validate_every > 0 and (i + 1) % config.validate_every == 0:
                    perplexity, _ = run_epoch(session, eval_mle_model, eval_gan_model,
                                              reader.validation(), config, vocab, None, 0, -1,
                                              gen_samples=config.gen_samples)
                    print "Epoch: %d Validation Perplexity: %.3f" % (i + 1, perplexity)
                    valid_perps.append(perplexity)
                else:
                    valid_perps.append(None)
                print 'Train:', train_perps
                print 'Valid:', valid_perps
                if steps >= config.max_steps:
                    break
        else:
            print '\nTesting'
            perplexity, _ = run_epoch(session, test_mle_model, test_gan_model, reader.testing(),
                                      config, vocab, None, 0, config.max_steps,
                                      gen_samples=config.gan_samples)
            print "Test Perplexity: %.3f" % perplexity


if __name__ == "__main__":
    tf.app.run()
