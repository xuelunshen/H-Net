import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../sean_utils'))
from ops import conv2d, dense

def predict(pfm, ufm, is_training, keep_prob):
    fm = tf.concat([pfm, ufm], axis = 3) # 7*7*512

    network= conv2d(fm, [2, 2, 512, 1024], [1, 1, 1, 1], is_training, bn=True,  scope='conv1_1')
    network = tf.nn.max_pool(network, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1') # [3, 3, 1024]
    
    network= conv2d(network, [2, 2, 1024, 2048], [1, 1, 1, 1], is_training, bn=True,  scope='conv2_1')
    network = tf.nn.max_pool(network, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2') # [32, 1, 1, 2048]

    fea_vec = tf.reshape(network, [-1, 2048])

    network = tf.nn.dropout(dense(fea_vec, 2048, 256, is_training, bn = True, scope = 'mlp1'), keep_prob)
    network = tf.nn.dropout(dense(network, 256, 128, is_training, bn = True, scope = 'mlp2'), keep_prob)
    network = tf.nn.dropout(dense(network, 128, 64, is_training, bn = True, scope = 'mlp3'), keep_prob)
    network = tf.nn.dropout(dense(network, 64, 1, is_training, activation_fn = tf.nn.tanh, bn = True, scope = 'mlp4'), keep_prob)

    return tf.reshape(network, [-1])