import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../sean_utils'))
from ops import conv2d_transpose

def decode(fm, training):
    fm = tf.reshape(fm, [-1, 2, 2, 256])
    network = conv2d_transpose('dcnv1', fm, 128, training, bn = False)
    network = conv2d_transpose('dcnv2', network, 64, training, bn = False)
    network = conv2d_transpose('dcnv3', network, 32, training, bn = False)
    network = conv2d_transpose('dcnv4', network, 16, training, bn = False)
    network = conv2d_transpose('dcnv5', network, 8, training, bn = False)
    network = conv2d_transpose('dcnv6', network, 4, training, bn = False)
    network = conv2d_transpose('dcnv7', network, 3, training, activation_fn = tf.nn.sigmoid, bn = False)

    return network