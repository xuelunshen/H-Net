import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../sean_utils'))
from ops import conv2d

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def encode(images, training):
    network= conv2d(images, [11, 11, 3, 96], [1, 4, 4, 1], training, bn=True, scope= 'conv1_1')
    network = tf.nn.max_pool(network, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
    
    network= conv2d(network, [5, 5, 96, 256], [1, 1, 1, 1], training, bn=True, scope= 'conv2_1')
    network = tf.nn.max_pool(network, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2')
    
    network= conv2d(network, [3, 3, 256, 384], [1, 1, 1, 1], training, bn=True, scope= 'conv3_1')
    network= conv2d(network, [3, 3, 384, 384], [1, 1, 1, 1], training, bn=True, scope= 'conv3_2')
    network= conv2d(network, [3, 3, 384, 256], [1, 1, 1, 1], training, bn=True, scope= 'conv3_3')
    network = tf.nn.max_pool(network, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool3')
    print_activations(network)
        
    return network
