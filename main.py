from __future__ import division

import sys
import time

import numpy as np
import tensorflow as tf

from config import cfg
from encdec import EncoderDecoderModel
from reader import Reader, Vocab
import utils


def call_mle_session(session, model, batch, use_gan, get_latent=False, encoder_only=False):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.data: batch[0],
              model.data_dropped: batch[1],
              model.lengths: batch[2]}
    ops = [model.nll, model.mle_cost]
    # training ops are tf.no_op() for a non-training model
    if encoder_only:
        train_ops = [model.mle_encoder_train_op]
    else:
        train_ops = [model.mle_train_op]
    if use_gan:
        ops.append(model.gan_cost)
        train_ops.append(model.d_train_op)
    if get_latent:
        ops.append(model.latent)
    ops.extend(train_ops)
    return session.run(ops, f_dict)[:-len(train_ops)]


def get_random_sample(random_dims):
    '''Generate a random latent representation to generate text from.'''
    return np.random.normal(size=random_dims)


def call_gan_session(session, model, random_dims, generator=False):
    '''Use the session to train the generator of the GAN with fake samples.'''
    f_dict = {model.rand_input: get_random_sample(random_dims)}
    ops = [model.gan_cost]
    # train_ops will be tf.no_op() for a non-training model
    if generator:
        ops.append(model.g_train_op)
    else:
        ops.append(model.d_train_op)
    return session.run(ops, f_dict)[:-1]


def display_sentences(output, vocab):
    '''Display sentences from indices.'''
    for i, sent in enumerate(output):
        print 'Sentence %d:' % i,
        words = []
        for word in sent:
            if word == vocab.eos_index:
                break
            words.append(vocab.vocab[word])
        if cfg.char_model:
            print ''.join(words)
        else:
            print ' '.join(words)
        print
    print


def generate_sentences(session, model, random_dims, vocab, mle_generator=False, true_output=None):
    '''Generate sentences using the generator, either novel or from known encodings (mle_generator).
    '''
    if mle_generator:
        print '\nTrue output'
        display_sentences(true_output[:, 1:], vocab)
        print 'Sentences generated from true encodings'
        f_dict = {model.latent: random_dims}
    else:
        print '\nNovel sentences: new batch'
        f_dict = {model.rand_input: get_random_sample(random_dims)}
    output = session.run(model.generated, f_dict)
    display_sentences(output, vocab)


def save_model(session, saver, perp, cur_iters):
    '''Save model file.'''
    save_file = cfg.save_file
    if not cfg.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print "Saving model (epoch perplexity: %.3f) ..." % perp
    save_file = saver.save(session, save_file)
    print "Saved to", save_file


def run_epoch(epoch, session, mle_model, gan_model, mle_generator, batch_loader, vocab,
              saver, steps, max_steps, gen_every=0, lr_tracker=None):
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
    shortterm_steps = 0
    nogan_steps = 0
    g_steps = 0
    d_steps = 0
    latest_latent = None
    scheduler = utils.Scheduler(cfg.min_d_acc, cfg.max_d_acc, cfg.max_perplexity,
                                cfg.sc_list_size, cfg.sc_decay)
    update_d = False
    update_g = False

    for step, batch in enumerate(batch_loader):
        if gen_every > 0 and (step + 1) % gen_every == 0:
            get_latent = True
        else:
            get_latent = False

        if lr_tracker is not None:
            lr_tracker.mle_mode()
        ret = call_mle_session(session, mle_model, batch, use_gan=update_d,
                               get_latent=get_latent)
        nll, mle_cost = ret[:2]

        if cfg.char_model:
            n_words = max(int((np.sum(batch[0] == vocab.vocab_lookup[' ']) / cfg.batch_size) + 1),
                          1)
        else:
            n_words = batch[0].shape[1] - 1
        scheduler.add_perp(np.exp(nll / n_words))
        update_d = scheduler.update_d()
        update_g = scheduler.update_g()

        if update_d:
            d_cost = ret[2]
        else:
            d_cost = -1.0
        if get_latent:
            latest_latent = ret[-1]
        if update_d:
            d_steps += 1
            r_cost = call_gan_session(session, gan_model,
                                      [cfg.batch_size, cfg.hidden_size])[0]
        else:
            r_cost = -1.0
        if update_g:
            g_steps += 1
            if lr_tracker is not None:
                lr_tracker.gan_mode()
            g_cost = call_gan_session(session, gan_model,
                                      [cfg.batch_size, cfg.hidden_size], generator=True)[0]
            if cfg.encoder_after_gan:  # FIXME this is weird, find a better alternative.
                ret = call_mle_session(session, mle_model, batch, use_gan=False, encoder_only=True)
                nll = (nll + ret[0]) / 2
                mle_cost = (mle_cost + ret[1]) / 2
        else:
            g_cost = -1.0
        costs = [c for c in [d_cost, r_cost, g_cost] if c > 0.0]
        if not costs:
            nogan_steps += 1
            gan_cost = 0.0
        else:
            gan_cost = np.mean(costs)
            d_acc = np.exp(-gan_cost)
            scheduler.add_d_acc(d_acc)

        nlls += nll
        mle_costs += mle_cost
        gan_costs += gan_cost
        shortterm_nlls += nll
        shortterm_mle_costs += mle_cost
        shortterm_gan_costs += gan_cost
        iters += n_words
        shortterm_iters += n_words
        shortterm_steps += 1

        if step % cfg.print_every == 0:
            avg_nll = shortterm_nlls / shortterm_iters
            avg_mle_cost = shortterm_mle_costs / shortterm_steps
            if shortterm_steps > nogan_steps:
                avg_gan_cost = shortterm_gan_costs / (shortterm_steps - nogan_steps)
                d_acc = np.exp(-avg_gan_cost)
            else:
                avg_gan_cost = -1.0
                d_acc = 0.0
            print("%d: %d  perplexity: %.3f  mle_loss: %.4f  mle_cost: %.4f  gan_cost: %.4f  "
                  "d_acc: %.4f  speed: %.0f wps  D%d G%d" %
                  (epoch + 1, step, np.exp(avg_nll), avg_nll, avg_mle_cost, avg_gan_cost, d_acc,
                   shortterm_iters * cfg.batch_size / (time.time() - start_time), d_steps, g_steps))

            shortterm_nlls = 0.0
            shortterm_mle_costs = 0.0
            shortterm_gan_costs = 0.0
            shortterm_iters = 0
            shortterm_steps = 0
            nogan_steps = 0
            g_steps = 0
            d_steps = 0
            start_time = time.time()

        if gen_every > 0 and (step + 1) % gen_every == 0:
            if latest_latent is not None:
                generate_sentences(session, mle_generator, latest_latent, vocab, True, batch[0])
            for _ in xrange(cfg.gen_samples):
                generate_sentences(session, gan_model, [cfg.batch_size, cfg.hidden_size],
                                   vocab)

        cur_iters = steps + step
        if saver is not None and cur_iters and cfg.save_every > 0 and \
                cur_iters % cfg.save_every == 0:
            save_model(session, saver, np.exp(nlls / iters), cur_iters)

        if max_steps > 0 and cur_iters >= max_steps:
            break

    if gen_every < 0:
        for _ in xrange(cfg.gen_samples):
            generate_sentences(session, gan_model, [cfg.batch_size, cfg.hidden_size], vocab)

    perp = np.exp(nlls / iters)
    cur_iters = steps + step
    if saver is not None and cfg.save_every < 0:
        save_model(session, saver, perp, cur_iters)
    return perp, cur_iters


def main(_):
    vocab = Vocab()
    vocab.load_from_pickle()
    reader = Reader(vocab)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        with tf.variable_scope("Model"):
            if cfg.training:
                with tf.variable_scope("LR"):
                    g_lr = tf.get_variable("g_lr", shape=[], initializer=tf.zeros_initializer,
                                           trainable=False)
                    d_lr = tf.get_variable("d_lr", shape=[], initializer=tf.zeros_initializer,
                                           trainable=False)
                lr_tracker = utils.LearningRateTracker(session, g_lr, d_lr)
                g_optimizer = utils.get_optimizer(g_lr, cfg.g_optimizer)
                d_optimizer = utils.get_optimizer(d_lr, cfg.d_optimizer)
                mle_model = EncoderDecoderModel(vocab, True, True, None, None,
                                                g_optimizer=g_optimizer, d_optimizer=d_optimizer)
                gan_model = EncoderDecoderModel(vocab, True, False, True, None,
                                                g_optimizer=g_optimizer, d_optimizer=d_optimizer)
                eval_mle_model = EncoderDecoderModel(vocab, False, True, True, None)
                eval_gan_model = EncoderDecoderModel(vocab, False, False, True, True)
            else:
                test_mle_model = EncoderDecoderModel(vocab, False, True, None, None)
                test_gan_model = EncoderDecoderModel(vocab, False, False, True, None)
            mle_generator = EncoderDecoderModel(vocab, False, False, True, True,
                                                mle_generator=True)
        saver = tf.train.Saver()
        try:
            # try to restore a saved model file
            saver.restore(session, cfg.load_file)
            print "Model restored from", cfg.load_file
        except ValueError:
            if cfg.training:
                tf.initialize_all_variables().run()
                print "No loadable model file, new model initialized."
            else:
                print "You need to provide a valid model file for testing!"
                sys.exit(1)

        if cfg.training:
            steps = 0
            train_perps = []
            valid_perps = []
            lr_tracker.update_mle_lr(cfg.mle_learning_rate)
            lr_tracker.update_g_lr(cfg.g_learning_rate)
            lr_tracker.update_d_lr(cfg.d_learning_rate)
            for i in xrange(cfg.max_epoch):
                print "\nEpoch: %d" % (i + 1)
                perplexity, steps = run_epoch(i, session, mle_model, gan_model, mle_generator,
                                              reader.training(), vocab, saver, steps,
                                              cfg.max_steps, gen_every=cfg.gen_every,
                                              lr_tracker=lr_tracker)
                print "Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity)
                train_perps.append(perplexity)
                if cfg.validate_every > 0 and (i + 1) % cfg.validate_every == 0:
                    perplexity, _ = run_epoch(i, session, eval_mle_model, eval_gan_model,
                                              mle_generator, reader.validation(), vocab,
                                              None, 0, -1, gen_every=-1)
                    print "Epoch: %d Validation Perplexity: %.3f" % (i + 1, perplexity)
                    valid_perps.append(perplexity)
                else:
                    valid_perps.append(None)
                print 'Train:', train_perps
                print 'Valid:', valid_perps
                if steps >= cfg.max_steps:
                    break
        else:
            print '\nTesting'
            perplexity, _ = run_epoch(0, session, test_mle_model, test_gan_model, mle_generator,
                                      reader.testing(), vocab, None, 0, cfg.max_steps,
                                      gen_every=-1)
            print "Test Perplexity: %.3f" % perplexity


if __name__ == "__main__":
    tf.app.run()
