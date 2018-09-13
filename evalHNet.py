import os
import sys
import time
import argparse

import tensorflow as tf

program_start_time = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../sean_utils/'))
data_base_path = os.path.join(BASE_DIR, '../../data/')

from io_tfrecord import tfio
from sean_functools import sess_config, get_learning_rate, str2bool

######################################### super parameters #########################################
parser = argparse.ArgumentParser()
parser.add_argument('--mpx',            default = 'val',    type = str,     help = 'model prefix : val [default: val]')
parser.add_argument('--logdir',         default = 'LOG/',   type = str,     help = 'log dir [default: LOG]')
parser.add_argument('--version',        default = 'v1',     type = str,     help = 'Model Version Name for Parallels Adjust Parameters [default: v1]')
parser.add_argument('--print_freq',     default = 125,      type = int,     help = 'Print Frequent [default: 125]')
parser.add_argument('--bgepoch',        default = 0,        type = int,     help = 'begin epoch [default: 0]')
parser.add_argument('--nepoch',         default = 100,      type = int,     help = 'number of train epoch [default: 100]')

parser.add_argument('--cuda_dev',       default = '0',      type = str,     help = 'cuda visible devices [default: 0]')
parser.add_argument('--gpu',            default = 1.0,      type = float,   help = 'GPU Proportion [default: 1.0]')

parser.add_argument('--batch_size',     default = 32,       type = int,     help = 'Batch Size [default: 32]')
parser.add_argument('--data_aug',       default = False,    type = str2bool,help = 'Use data augment [default: False]')
parser.add_argument('--vsize',          default = 1024,     type = int,     help = 'Feature Vector Dims [default: 1024]')

parser.add_argument('--lr',             default = 0.001,    type = float,   help = 'Initial learning rate [default: 0.001]')
parser.add_argument('--lrb',            default = 0.0002,   type = float,   help = 'learning rate base line [default: 0.0002]')
parser.add_argument('--optimizer',      default = 'AMSGrad',type = str,     help = 'adam || RMSProp || AMSGrad || Momentum [default: AMSGrad]')
parser.add_argument('--momentum',       default = 0.9,      type = float,   help = 'momentum for optimizer momentum [default: 0.9]')
parser.add_argument('--decay_step',     default = 1,        type = int,     help = 'Decay step for lr decay in multi epoch [default: 1 epoch decay onece]')
parser.add_argument('--decay_rate',     default = 0.9,      type = float,   help = 'Decay rate for lr decay in each decay_step [default: 0.9]')

FLAGS = parser.parse_args()

# super parameter parse
# MPX = FLAGS.mpx
# LOGDIR = FLAGS.logdir
# VERSION = FLAGS.version
# PRINT_FREQ = FLAGS.print_freq
# BGEPOCH = FLAGS.bgepoch
# NEPOCH = FLAGS.nepoch

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_dev
GPU = FLAGS.gpu

BATCH_SIZE = FLAGS.batch_size
VSIZE = FLAGS.vsize

BASE_LR = FLAGS.lr
DATA_AUG = FLAGS.data_aug
LR_BASElINE = FLAGS.lrb
OPTIMIZER = FLAGS.optimizer
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
######################################### super parameters #########################################

# log prepare
# if not os.path.exists(LOGDIR): os.mkdir(LOGDIR)
# NET_NAME = BASE_DIR.split('\\')[-1] if BASE_DIR.find('\\') != -1 else BASE_DIR.split('/')[-1]
# LOG_FOUT = open((LOGDIR + NET_NAME + '-' + VERSION + '.txt'), 'w')
def logout(out_str):
    # LOG_FOUT.write(out_str + '\n')
    # LOG_FOUT.flush()
    print(out_str)

# placeholder
config = sess_config(GPU)
width, height, channel = 256, 256, 3
photo_pl = tf.placeholder(tf.float32, [None, width, height, channel], name = 'photo_pl')
photo_flat = tf.reshape(photo_pl, [-1, width * height * channel], name = 'photo_flat')
uds_pl = tf.placeholder(tf.float32, [None, width, height, channel], name = 'uds_pl')
uds_flat = tf.reshape(uds_pl, [-1, width * height * channel], name = 'uds_flat')
label_pl = tf.placeholder(tf.int32, name = 'label_pl')
keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob_pl')
is_training_pl = tf.placeholder(tf.bool, name = 'is_training_pl')

# data io
train_file = data_base_path + 'H-Net_Data_lwq/test/'
valid_file = data_base_path + 'H-Net_Data_lwq/tt2/'
test_file = data_base_path + 'H-Net_Data_lwq/test/'
io = tfio(BATCH_SIZE, 1000, 0, False, width, height, channel, train_file, valid_file, test_file, 1001, 1)

# train strategy
global_step = tf.Variable(0)
one_epoch_step = io.get_train_itr()
learning_rate = get_learning_rate(global_step, one_epoch_step, BASE_LR, DECAY_STEP, DECAY_RATE, LR_BASElINE)

# network prepare
from H_Net import H_Net
network = H_Net(photo_pl, photo_flat, uds_pl, uds_flat, label_pl, is_training_pl, keep_prob_pl, VSIZE, OPTIMIZER, MOMENTUM, learning_rate, global_step, logout)

# trainable variables contain batch normalization mean and variance
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars

# output parameters
keys = sorted(vars(FLAGS).keys())
for key in keys:
    if len(key) < 4: k = key + '\t\t\t'
    elif len(key) < 8: k = key + '\t\t'
    elif len(key) < 12: k = key + '\t'
    else: k = key
    logout(k + '\t:' + str(getattr(FLAGS, key)))

def eval_net(sess, batch_num, batch, batch_size, acc, _predict):
    TP, FP, TN, FN = 0, 0, 0, 0
    val_acc, n_batch = 0, 0
    for index in range(batch_num):
        p, u, l = sess.run(batch)
        ac, r = sess.run([acc, _predict], feed_dict = {photo_pl:p, uds_pl:u, label_pl:l, keep_prob_pl:1.0, is_training_pl:False})
        print('%d/%d\t%f' % (index, batch_num, ac))
        val_acc += ac
        n_batch += 1
        
        for i in range(batch_size):
            if 1 == l[i] == r[i]: TP = TP + 1
            if -1 == l[i] and 1 == r[i]: FP = FP + 1
            if -1 == l[i] == r[i]: TN = TN + 1
            if 1 == l[i] and -1 == r[i]: FN = FN + 1
    latest_val_acc = val_acc / n_batch
    logout('TP:{:d}\tFP:{:d}\tTN:{:d}\tFN:{:d} AC:{:f}'.format(TP, FP, TN, FN, latest_val_acc))
    return latest_val_acc

def eval_network():
    valid_p_batch, valid_u_batch, valid_l_batch = io.get_onebatch_validdata(width = width, height = height)
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
        model_path = os.path.join(BASE_DIR, 'model')
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            model_saver = tf.train.Saver(var_list = var_list)
            model_saver.restore(sess, ckpt.model_checkpoint_path)
            print(('\n*** %s model is loaded ***\n' % ckpt.model_checkpoint_path))
        else:
            print('load model %s fail and exit' % model_path)
            exit(-1)
        
        print('valid files\t\t:{:d}'.format(io.valid_num))
        
        test_start_time = time.time()
        eval_net(sess, io.get_valid_itr(), [valid_p_batch, valid_u_batch, valid_l_batch], io.batch_size_v, network.metricacc, network.predict)
        print("test tooks %f min" % ((time.time() - test_start_time) / 60))
        
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    eval_network()
    logout('cost: {:f} seconds'.format((time.time() - program_start_time)))