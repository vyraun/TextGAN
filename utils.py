import itertools
import re

import nltk
import numpy as np
import tensorflow as tf


class Scheduler(object):

    '''Scheduler for GANs'''

    def __init__(self, min_d_acc, max_d_acc, max_perp, list_size, decay):
        self.min_d_acc = min_d_acc
        self.max_d_acc = max_d_acc
        self.max_perp = max_perp
        self.list_size = list_size
        self.d_accs = []
        self.perps = []
        coeffs = [1.0]
        for _ in xrange(list_size - 1):
            coeffs.append(coeffs[-1] * decay)
        self.coeffs = np.array(coeffs) / sum(coeffs)

    def add_d_acc(self, d_acc):
        '''Observe new descriminator accuracy.'''
        self.d_accs.insert(0, d_acc)
        if len(self.d_accs) > self.list_size:
            self.d_accs.pop()

    def add_perp(self, perp):
        '''Observe new perplexity.'''
        self.perps.insert(0, perp)
        if len(self.perps) > self.list_size:
            self.perps.pop()

    def _current_perp(self):
        '''Smooth approximation of current perplexity.'''
        coeffs = self.coeffs.copy(order='K')
        if len(self.perps) < self.list_size:
            coeffs = coeffs[:len(self.perps)]
            coeffs /= np.sum(coeffs)
        return np.sum(np.array(self.perps) * coeffs)

    def _current_d_acc(self):
        '''Smooth approximation of current descriminator accuracy.'''
        coeffs = self.coeffs.copy(order='K')
        if len(self.d_accs) < self.list_size:
            coeffs = coeffs[:len(self.d_accs)]
            coeffs /= np.sum(coeffs)
        return np.sum(np.array(self.d_accs) * coeffs)

    def update_d(self):
        '''Whether or not to update the descriminator.'''
        if len(self.perps) < self.list_size or (self.max_perp > 0.0 and
                                                self._current_perp() > self.max_perp):
            return False
        if len(self.d_accs) == 0 or self._current_d_acc() < self.max_d_acc:
            return True
        else:
            return False

    def update_g(self):
        '''Whether or not to update the generator.'''
        if len(self.perps) < self.list_size or (self.max_perp > 0.0 and
                                                self._current_perp() > self.max_perp):
            return False
        if len(self.d_accs) > 0 and self._current_d_acc() > self.min_d_acc:
            return True
        else:
            return False


class LearningRateTracker(object):

    '''Keep track of the current learning rates so as to only do session updates when necessary.'''

    def __init__(self, session, g_lr, d_lr, g_mle=True):
        self.session = session
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.mle_lr_value = 0.
        self.g_lr_value = 0.
        self.d_lr_value = 0.
        self.g_mle = g_mle

    def _set_g_lr(self, value):
        self.session.run(tf.assign(self.g_lr, value))

    def _set_d_lr(self, value):
        self.session.run(tf.assign(self.d_lr, value))

    def update_mle_lr(self, value):
        if self.mle_lr_value != value:
            print 'Changing MLE learning rate to', value
            if self.g_mle:
                self._set_g_lr(value)
            self.mle_lr_value = value

    def update_g_lr(self, value):
        if self.g_lr_value != value:
            print 'Changing generator learning rate to', value
            if not self.g_mle:
                self._set_g_lr(value)
            self.g_lr_value = value

    def update_d_lr(self, value):
        if self.d_lr_value != value:
            print 'Changing discriminator learning rate to', value
            self._set_d_lr(value)
            self.d_lr_value = value

    def mle_mode(self):
        if not self.g_mle:
            self.g_mle = True
            if self.g_lr_value != self.mle_lr_value:
                self._set_g_lr(self.mle_lr_value)

    def gan_mode(self):
        if self.g_mle:
            self.g_mle = False
            if self.g_lr_value != self.mle_lr_value:
                self._set_g_lr(self.g_lr_value)


fix_re = re.compile(r'''[^a-z0-9"'?.,]+''')
num_re = re.compile(r'[0-9]+')


def fix_word(word):
    word = word.lower()
    word = fix_re.sub('', word)
    word = num_re.sub('#', word)
    return word


def read_words(line, chars):
    if chars:
        first = True
    # workaround to get the NLTK tokenization deal with <unk> nicely
    for raw_word in nltk.word_tokenize(line.replace('<unk>', '-unk-')):
        if raw_word == '-unk-':
            word = '<unk>'
        else:
            word = fix_word(raw_word)
        if word:
            if chars:
                if not first:
                    yield ' '
                else:
                    first = False
                if word == '<unk>':
                    yield word
                else:
                    for c in word:
                        yield c
            else:
                yield word


def grouper(n, iterable, fillvalue=None):
    '''Group elements of iterable in groups of n. For example:
       >>> [e for e in grouper(3, [1,2,3,4,5,6,7])]
       [(1, 2, 3), (4, 5, 6), (7, None, None)]'''
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def get_optimizer(lr, name):
    '''Return an optimizer.'''
    if name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif name == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr)
    elif name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(lr)
    return optimizer


def rowwise_lookup(params, indices):
    '''Look up an index from each row of params as per indices.'''
    shape = params.get_shape().as_list()
    if len(shape) == 2:
        hidden_size = 1
    else:
        hidden_size = shape[-1]
    flattened = tf.reshape(params, [-1, hidden_size])
    flattened_indices = indices + (tf.range(shape[0]) * tf.shape(params)[1])
    return tf.gather(flattened, flattened_indices)


def linear(args, output_size, bias, bias_start=0.0, scope=None, initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    Based on the code from TensorFlow."""
    if not tf.nn.nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype,
                                 initializer=initializer)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                    initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term


def highway(input_, layer_size=1, bias=-2, f=tf.nn.tanh):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate."""
    if tf.nn.nest.is_sequence(input_):
        input_ = tf.concat(1, input_)
    shape = input_.get_shape()
    if len(shape) != 2:
        raise ValueError("Highway is expecting 2D arguments: %s" % str(shape))
    size = shape[1]
    for idx in xrange(layer_size):
        output = f(linear(input_, size, False, scope='Highway_Nonlin_%d' % idx))
        transform_gate = tf.sigmoid(linear(input_, size, False, scope='Highway_Gate_%d' % idx)
                                    + bias)
        carry_gate = 1.0 - transform_gate
        output = transform_gate * output + carry_gate * input_
        input_ = output

    return output
