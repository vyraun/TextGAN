import itertools
import re

import nltk
import numpy as np
import tensorflow as tf


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_color(s, color=None):
    if color:
        print color + str(s) + Colors.ENDC,
    else:
        print s,


fix_re = re.compile(r'''[^a-z0-9"'?.,]+''')
num_re = re.compile(r'[0-9]+')

def fix_word(word):
    word = word.lower()
    word = fix_re.sub('', word)
    word = num_re.sub('#', word)
    return word


class Scheduler(object):
    '''Scheduler for GANs'''
    def __init__(self, min_d_acc, max_d_acc, max_perp, list_size=6, decay=0.4):
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
        return np.sum(np.array(self.perps) * self.coeffs)

    def _current_d_acc(self):
        '''Smooth approximation of current descriminator accuracy.'''
        return np.sum(np.array(self.d_accs) * self.coeffs)

    def update_d(self):
        '''Whether or not to update the descriminator.'''
        if len(self.perps) < self.list_size or self._current_perp() > self.max_perp:
            return False
        if len(self.d_accs) < self.list_size or self._current_d_acc() < self.max_d_acc:
            return True
        else:
            return False

    def update_g(self):
        '''Whether or not to update the generator.'''
        if len(self.perps) < self.list_size or self._current_perp() > self.max_perp:
            return False
        if len(self.d_accs) == self.list_size and self._current_d_acc() > self.min_d_acc:
            return True
        else:
            return False


def read_words(line):
    # workaround to get the NLTK tokenization deal with <unk> nicely
    for raw_word in nltk.word_tokenize(line.replace('<unk>', '-unk-')):
        if raw_word == '-unk-':
            yield '<unk>'
        else:
            word = fix_word(raw_word)
            if word:
                yield word


def grouper(n, iterable, fillvalue=None):
    '''Group elements of iterable in groups of n. For example:
       >>> [e for e in grouper(3, [1,2,3,4,5,6,7])]
       [(1, 2, 3), (4, 5, 6), (7, None, None)]'''
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def get_optimizer(config, lr):
    '''Return an optimizer based on the configuration'''
    if config.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif config.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr)
    elif config.optimizer == 'adadelta':
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
        initializer=tf.contrib.layers.xavier_initializer(uniform=False)
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
        transform_gate = tf.sigmoid(linear(input_, size, False, scope='Highway_Gate_%d' % idx) \
                                    + bias)
        carry_gate = 1.0 - transform_gate
        output = transform_gate * output + carry_gate * input_
        input_ = output

    return output
