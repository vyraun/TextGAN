from __future__ import division

import sys
import time

import numpy as np
import tensorflow as tf

from config import Config
from encdec import EncoderDecoderModel
from reader import Reader, Vocab


def call_mle_session(session, model, batch, use_gan, get_latent=False):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.ldata: batch[0],
              model.rdata: batch[1],
              model.ldata_dropped: batch[2],
              model.rdata_dropped: batch[3],
              model.lengths: batch[4]}
    ops = [model.nll, model.mle_cost]
    train_ops = [model.mle_train_op] # this will be tf.no_op() for a non-training model
    if use_gan:
        ops.append(model.gan_cost)
        train_ops.append(model.d_train_op) # tf.no_op() for non-training model
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
        for word in sent:
            if word == vocab.eos_index:
                break
            print vocab.vocab[word],
        print
    print


def generate_sentences(session, model, random_dims, vocab, mle_generator=False, true_output=None):
    '''Generate sentences using the generator, either novel or from known encodings (mle_generator).
    '''
    if mle_generator:
        print '\nTrue output'
        display_sentences(true_output[:,1:], vocab)
        print 'Sentences generated from true encodings'
        f_dict = {model.latent: random_dims}
    else:
        print '\nNovel sentences: new batch'
        f_dict = {model.rand_input: get_random_sample(random_dims)}
    output = session.run(model.generated, f_dict)
    display_sentences(output, vocab)


def save_model(session, saver, config, perp, cur_iters):
    '''Save model file.'''
    save_file = config.save_file
    if not config.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print "Saving model (epoch perplexity: %.3f) ..." % perp
    save_file = saver.save(session, save_file)
    print "Saved to", save_file


def run_epoch(epoch, session, mle_model, gan_model, mle_generator, batch_loader, config, vocab,
              saver, steps, max_steps, gen_every=0, use_gan=True):
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
    latest_latent = None

    for step, batch in enumerate(batch_loader):
        if gen_every > 0 and (step + 1) % gen_every == 0:
            get_latent = True
        else:
            get_latent = False
        if use_gan:
            latest_count += 1
            ret = call_mle_session(session, mle_model, batch, use_gan=True, get_latent=get_latent)
            nll, mle_cost, d_cost = ret[:3]
            if get_latent:
                latest_latent = ret[-1]
            r_cost = call_gan_session(session, gan_model, [config.batch_size, config.hidden_size])
            gan_cost = (d_cost + r_cost) / 2
            if (step + 1) % config.update_g_every == 0:
                g_cost = call_gan_session(session, gan_model,
                                          [config.batch_size, config.hidden_size], generator=True)
                gan_cost = (d_cost + r_cost + g_cost) / 3
        else:
            ret = call_mle_session(session, mle_model, batch, use_gan=False, get_latent=get_latent)
            nll, mle_cost = ret[:2]
            if get_latent:
                latest_latent = ret[-1]
            gan_cost = 0.0

        nlls += nll
        mle_costs += mle_cost
        gan_costs += gan_cost
        shortterm_nlls += nll
        shortterm_mle_costs += mle_cost
        shortterm_gan_costs += gan_cost
        # batch[1] is the right aligned batch, without <eos>. predictions also have one token less
        # (no <sos>).
        iters += batch[1].shape[1]
        shortterm_iters += batch[1].shape[1]
        shortterm_steps += 1

        if step % config.print_every == 0:
            avg_nll = shortterm_nlls / shortterm_iters
            avg_mle_cost = shortterm_mle_costs / shortterm_steps
            avg_gan_cost = shortterm_gan_costs / shortterm_steps
            d_acc = np.exp(-avg_gan_cost)
            print("%d : %d  perplexity: %.3f  mle_loss: %.4f  mle_cost: %.4f  gan_cost: %.4f  "
                  "d_acc: %.4f  speed: %.0f wps" %
                  (epoch+1, step, np.exp(avg_nll), avg_nll, avg_mle_cost, avg_gan_cost, d_acc,
                   shortterm_iters * config.batch_size / (time.time() - start_time)))

            shortterm_nlls = 0.0
            shortterm_mle_costs = 0.0
            shortterm_gan_costs = 0.0
            shortterm_iters = 0
            shortterm_steps = 0
            start_time = time.time()

        if gen_every > 0 and (step + 1) % gen_every == 0:
            if latest_latent is not None:
                generate_sentences(session, mle_generator, latest_latent, vocab, True, batch[0])
            if use_gan:
                for _ in xrange(config.gen_samples):
                    generate_sentences(session, gan_model, [config.batch_size, config.hidden_size],
                                       vocab)

        cur_iters = steps + step
        if saver is not None and cur_iters and config.save_every > 0 and \
                cur_iters % config.save_every == 0:
            save_model(session, saver, config, np.exp(nlls / iters), cur_iters)

        if max_steps > 0 and cur_iters >= max_steps:
            break

    if use_gan and gen_every < 0:
        for _ in xrange(config.gen_samples):
            generate_sentences(session, gan_model, [config.batch_size, config.hidden_size], vocab)

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
        with tf.variable_scope("Model"):
            if config.training:
                mle_model = EncoderDecoderModel(config, vocab, True, True, None, None)
                gan_model = EncoderDecoderModel(config, vocab, True, False, True, None)
                eval_mle_model = EncoderDecoderModel(config, vocab, False, True, True, None)
                eval_gan_model = EncoderDecoderModel(config, vocab, False, False, True, True)
            else:
                test_mle_model = EncoderDecoderModel(config, vocab, False, True, None, None)
                test_gan_model = EncoderDecoderModel(config, vocab, False, False, True, None)
            mle_generator = EncoderDecoderModel(config, vocab, False, False, True, True, True)
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
            use_gan = False
            for i in xrange(config.max_epoch):
                if i == config.gan_wait_epochs:
                    print 'Enabling GAN training!'
                    use_gan = True
                print "\nEpoch: %d MLE learning rate: %.4f, D learning rate: %.4f, " \
                      "G learning rate: %.4f" % (i + 1, session.run(mle_model.mle_lr),
                                           session.run(gan_model.d_lr), session.run(gan_model.g_lr))
                perplexity, steps = run_epoch(i, session, mle_model, gan_model, mle_generator,
                                              reader.training(), config, vocab, saver, steps,
                                              config.max_steps, gen_every=config.gen_every,
                                              use_gan=use_gan)
                print "Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity)
                train_perps.append(perplexity)
                if config.validate_every > 0 and (i + 1) % config.validate_every == 0:
                    perplexity, _ = run_epoch(i, session, eval_mle_model, eval_gan_model,
                                              mle_generator, reader.validation(), config, vocab,
                                              None, 0, -1, gen_every=-1, use_gan=use_gan)
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
            perplexity, _ = run_epoch(0, session, test_mle_model, test_gan_model, mle_generator,
                                      reader.testing(), config, vocab, None, 0, config.max_steps,
                                      gen_every=-1)
            print "Test Perplexity: %.3f" % perplexity


if __name__ == "__main__":
    tf.app.run()
