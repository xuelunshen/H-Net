import tensorflow as tf
from selu import selu, initializer

def conv2d(x, w_shape, strides, is_training, activation_fn = selu, bn = False, scope = 'conv', padding = 'SAME', w_initializer=initializer, b_initializer=initializer):
    with tf.variable_scope(scope):
        kernel = tf.get_variable(name = 'weights', shape = w_shape, dtype = tf.float32, initializer = w_initializer)
        bias = tf.get_variable(name = 'biases', shape = [w_shape[3]], dtype = tf.float32, trainable = True, initializer = b_initializer)
        outputs = tf.nn.conv2d(x, kernel, strides, padding = padding) + bias
        
        if bn:
            outputs = batch_norm_layer(outputs, is_training, 'batch_norm')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
        return outputs

def conv2d_transpose(name_or_scope, x, n_filters, training, k_h = 5, k_w = 5, stride_h = 2, stride_w = 2, padding = 'SAME', activation_fn = selu, bn = True):
    with tf.variable_scope(name_or_scope):
        static_input_shape = x.get_shape().as_list()
        dyn_input_shape = tf.shape(x)
        
        # extract batch-size like as a symbolic tensor to allow variable size
        batch_size = dyn_input_shape[0]
        
        w = tf.get_variable('weights', [k_h, k_w, n_filters, static_input_shape[3]], dtype = tf.float32, initializer = initializer)
        
        assert padding in {'SAME', 'VALID'}
        if padding is 'SAME':
            out_h = dyn_input_shape[1] * stride_h
            out_w = dyn_input_shape[2] * stride_w
        else:
            out_h = (dyn_input_shape[1] - 1) * stride_h + k_h
            out_w = (dyn_input_shape[2] - 1) * stride_w + k_w
        
        out_shape = tf.stack([batch_size, out_h, out_w, n_filters])
        
        convt = tf.nn.conv2d_transpose(x, w, output_shape = out_shape, strides = [1, stride_h, stride_w, 1], padding = padding)
        
        b = tf.get_variable('biases', [n_filters], dtype = tf.float32, initializer = initializer)
        outputs = convt + b
        
        if bn:
            outputs = batch_norm_layer(outputs, training, 'batch_norm')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    return outputs

def dense(x, inputFeatures, outputFeatures, is_training, activation_fn = selu, bn = False, scope = 'mlp'):
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('weights', [inputFeatures, outputFeatures], tf.float32, initializer)
        bias = tf.get_variable('biases', [outputFeatures], tf.float32, initializer = initializer)
        outputs = tf.matmul(x, matrix) + bias
        
        if bn:
            outputs = batch_norm_layer(outputs, is_training, 'batch_norm')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
        return outputs

def batch_norm_layer(x, train_phase, scope_bn):
    z = tf.layers.batch_normalization(x, training = train_phase)
    return z


