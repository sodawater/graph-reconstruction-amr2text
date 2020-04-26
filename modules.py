from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from linalg import LinearOperatorLowerTriangular
from tensorflow.python.keras import backend as K
import math

def index_matrix_to_pairs(index_matrix):
  # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
  #                        [[0, 2], [1, 3], [2, 1]]]
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1),
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld

def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.
  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
  Args:
    features: A `Tensor` representing preactivation values. Must be one of
      the following types: `float16`, `float32`, `float64`, `int32`, `int64`.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]) as name:
    features = ops.convert_to_tensor(features, name="features")
    if features.dtype.is_integer:
      features = math_ops.to_float(features)
    alpha = ops.convert_to_tensor(alpha, dtype=features.dtype, name="alpha")
    return math_ops.maximum(alpha * features, features, name=name)

def selu(x):
  """Scaled Exponential Linear Unit (SELU).
  SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
  are pre-defined constants. The values of `alpha` and `scale` are
  chosen so that the mean and variance of the inputs are preserved
  between two consecutive layers as long as the weights are initialized
  correctly (see `lecun_normal` initialization) and the number of inputs
  is "large enough" (see references for more information).
  Arguments:
      x: A tensor or variable to compute the activation function for.
  Returns:
      The scaled exponential unit activation: `scale * elu(x, alpha)`.
  # Note
      - To be used together with the initialization "lecun_normal".
      - To be used together with the dropout variant "AlphaDropout".
  References:
      - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * K.elu(x, alpha)

def norm_log_liklihood(x, mu, logvar):
    return -0.5*tf.reduce_sum(tf.log(2*np.pi) + logvar + tf.div(tf.pow((x-mu), 2), tf.exp(logvar)), reduction_indices=1)


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z


def reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return array_ops.reverse(input_, axis=[seq_dim])

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):

    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
    # return init_lr  * tf.minimum(step * warmup_steps ** -1.5, warmup_steps ** 1.5 * step ** -2.0)

def multihead_attention(queries,
                        keys,
                        query_length,
                        key_length,
                        values=None,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        pointer=False,
                        logits=False,
                        using_mask=False,
                        no_tile=False,
                        mymasks=None,
                        scope="multihead_attention",
                        no_att=False,
                        reuse=tf.AUTO_REUSE):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        l = tf.shape(queries)[1]
        #print(l)
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        if values==None:
            Q = tf.layers.dense(queries, num_units, activation=None, use_bias=False, name="q")  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None, use_bias=False, name="k")  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None, use_bias=False, name="v")  # (N, T_k, C)
        else:
            Q = tf.layers.dense(queries, num_units, activation=None, use_bias=False, name="q")  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None, use_bias=False, name="k")  # (N, T_k, C)
            V = tf.layers.dense(values, num_units, activation=None, use_bias=False, name="v")  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)



        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        if key_length != None:
            key_masks = tf.sequence_mask(key_length, tf.shape(keys)[1], dtype=tf.float32)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        if logits == True:
            return outputs

        if pointer == True:
            outputs = tf.nn.softmax(outputs)

            return outputs

        if using_mask:
            if not no_tile:
                mymask = tf.tile(mymasks, [num_heads, 1, 1])
            else:
                mymask = mymasks
            outputs = tf.where(tf.equal(mymask, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)


        # Query Masking
        # query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.sequence_mask(query_length, tf.shape(queries)[1], dtype=tf.float32)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        # query_masks = tf.where(tf.equal(query_masks, 0), paddings, tf.zeros_like(outputs, dtype=tf.float32))
        outputs *= query_masks


        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        weights = tf.reduce_mean(tf.concat(tf.split(tf.expand_dims(outputs,3), num_heads, axis=0), axis=3), axis=3)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        outputs = tf.layers.dense(tf.concat(tf.split(outputs, num_heads, axis=0), axis=2), num_units, activation=None,
                                  use_bias=False, name="concat")  # (N, T_q, C)
        if no_att:
            return outputs, weights
        return outputs


def multihead_attention_edge(queries,
                        keys,
                        edges,
                        pairs,
                        query_length,
                        key_length,
                        graph=False,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        pointer=False,
                        logits=False,
                        using_mask=False,
                        no_tile=False,
                        mymasks=None,
                        use_glu=False,
                        use_graph=False,
                        graph_masks=None,
                        coverage=False,
                        history=None,
                        length=None,
                        gate=None,
                        sigmoid=False,
                        gmode=None,
                        scope="multihead_attention",
                        no_att=False,
                        reuse=tf.AUTO_REUSE):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        l = tf.shape(queries)[1]
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        V = tf.layers.dense(keys, num_units, activation=None, use_bias=False, name="v")  # (N, T_k, C)

        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        P = tf.layers.dense(pairs, num_units * num_heads, activation=None, use_bias=False, name="P")
        P_ = tf.concat(tf.split(P, num_heads, axis=2), axis=0)

        outputs = tf.layers.dense(leaky_relu(P_), 1, use_bias=False)

        outputs = tf.transpose(tf.tile(outputs, [1, 1, l]), [0, 2, 1])
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        if key_length != None:
            key_masks = tf.sequence_mask(key_length, tf.shape(keys)[1], dtype=tf.float32)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            if no_att:
                ones = tf.ones_like(outputs)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, ones)
            else:
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        if logits == True:
            return outputs

        if pointer == True:
            outputs = tf.nn.softmax(outputs)

            return outputs

        if using_mask:
            if not no_tile:
                mymask = tf.tile(mymasks, [num_heads, 1, 1])
            else:
                mymask = mymasks
            outputs = tf.where(tf.equal(mymask, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)

        # Query Masking
        # query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.sequence_mask(query_length, tf.shape(queries)[1], dtype=tf.float32)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        # query_masks = tf.where(tf.equal(query_masks, 0), paddings, tf.zeros_like(outputs, dtype=tf.float32))
        outputs *= query_masks



        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        outputs = tf.layers.dense(tf.concat(tf.split(outputs, num_heads, axis=0), axis=2), num_units, activation=None,
                                  use_bias=False, name="concat")  # (N, T_q, C)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        return outputs

# def positional_encoding(inputs,
#                         batch_size,
#                         length,
#                         num_units,
#                         zero_pad=True,
#                         scale=True,
#                         scope="positional_encoding",
#                         reuse=None):
#     '''Sinusoidal Positional_Encoding.
#     Args:
#       inputs: A 2d Tensor with shape of (N, T).
#       num_units: Output dimensionality
#       zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
#       scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
#       scope: Optional scope for `variable_scope`.
#       reuse: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#     Returns:
#         A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
#     '''
#
#     # N, T, _ = inputs.get_shape().as_list()
#     N, T = batch_size, length
#     with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#         position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
#
#         # First part of the PE function: sin and cos argument
#         position_enc = np.array([
#             [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
#             for pos in range(T)], dtype=np.float32)
#
#         # Second part, apply the cosine to even columns and sin to odds.
#         position_enc[:, 0::2] = np.sin(position_enc[:, 0::2], dtype=np.float32)  # dim 2i
#         position_enc[:, 1::2] = np.cos(position_enc[:, 1::2], dtype=np.float32)  # dim 2i+1
#
#         # Convert to a tensor
#         lookup_table = tf.convert_to_tensor(position_enc)
#
#         if zero_pad:
#             lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
#                                       lookup_table[1:, :]), 0)
#         outputs = tf.nn.embedding_lookup(lookup_table, inputs)
#
#         if scale:
#             outputs = outputs * num_units**0.5
#
#         return outputs
def positional_encoding(length,
                        inputs,
                        num_units,
                        zero_pad=False,
                        scale=False,
                        time=-1,
                        input_pos=None,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''
    E = num_units # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    # print(N, T, E)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        position_enc = np.array([
            [pos * np.exp(-np.log(10000.0) * ((i * 1.0 - i % 2)/2) / (E * 1.0 /2)) for i in range(E)]
            for pos in range(length)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        if input_pos ==None:
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        else:
            outputs = tf.nn.embedding_lookup(position_enc, input_pos)
        # masks
        if zero_pad:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)



def feedforward(inputs,
                num_units=[2048, 512],
                scope="feedforward",
                is_training=False,
                dropout_rate=0,
                activation=None,
                reuse=tf.AUTO_REUSE):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        # params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
        #           "activation": gelu, "use_bias": True}
        if activation == None:
            activation = gelu
        outputs = tf.layers.dense(inputs, num_units[0], activation=activation)

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs