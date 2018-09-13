import AMSGrad
import tensorflow as tf

from network import encoder, decoder, metriclayer
from ops import conv2d
from ops import dense

from sean_functools import get_metricacc, genloss, hingeloss, get_var

keyname = {'pe':'photo_encoder', 'pd':'photo_decoder', 'ue':'uds_encoder', 'ud':'uds_decoder', 'ml':'metriclayer'}
ae_keyname = {'pe':'photo_encoder', 'pd':'photo_decoder', 'ue':'uds_encoder', 'ud':'uds_decoder'}
ml_keyname = {'ml':'metriclayer'}

class H_Net:
    def __init__(self, photo_pl, photo_flat, uds_pl, uds_flat, label_pl, training, keep_prob_pl, vsize, OPTIMIZER, MOMENTUM, learning_rate, global_step, logout):
        #photo_encoder
        with tf.variable_scope(keyname['pe']):
            pfm = encoder.encode(photo_pl, training)
            w = pfm.get_shape()[1].value
            h = pfm.get_shape()[2].value
            c = pfm.get_shape()[3].value
            network = conv2d(pfm, [w, h, c, vsize], [1, 1, 1, 1], training, bn = True, scope = 'con2vector', padding = 'VALID')
            pz_mean = tf.reshape(network, [-1, vsize])
    
        # uds_encoder
        with tf.variable_scope(keyname['ue']):
            ufm = encoder.encode(uds_pl, training)
            w = ufm.get_shape()[1].value
            h = ufm.get_shape()[2].value
            c = ufm.get_shape()[3].value
            network = conv2d(ufm, [w, h, c, vsize], [1, 1, 1, 1], training, bn = True, scope = 'conv2vector', padding = 'VALID')
            uz_mean = tf.reshape(network, [-1, vsize])
    
        # photo_decoder
        with tf.variable_scope(keyname['pd']):
            p_to_fm = pz_mean
            if vsize != 1024: p_to_fm = dense(p_to_fm, vsize, 2 * 2 * 256, training, bn = True, scope = 'mlp2vector')
            p_to_fm = tf.reshape(p_to_fm, [-1, 2, 2, 256])
            pgen_image = decoder.decode(p_to_fm, training)
    
        # photo_decoder_to_fm
        with tf.variable_scope(keyname['ud']):
            u_to_fm = uz_mean
            if vsize != 1024: u_to_fm = dense(u_to_fm, vsize, 2 * 2 * 256, training, bn = True, scope = 'mlp2vector')
            u_to_fm = tf.reshape(u_to_fm, [-1, 2, 2, 256])
            ugen_image = decoder.decode(u_to_fm, training)
    
        with tf.variable_scope(keyname['ml']):
            pred = metriclayer.predict(pfm, ufm, training, keep_prob_pl)
        
        #accuracy
        metricacc, predict = get_metricacc(pred, label_pl)
        self.metricacc = metricacc
        self.predict = predict
        
        # loss
        photogen_loss = genloss(pgen_image, photo_flat)
        udsvgen_loss = genloss(ugen_image, uds_flat)
        gen_loss = photogen_loss + udsvgen_loss
        self.gen_loss = gen_loss
        hinge_loss = hingeloss(label_pl, pred)
        self.hinge_loss = hinge_loss
        
        # var list
        ae_varlist, ae_wlist = get_var(logout, ae_keyname)
        ml_varlist, ml_wlist = get_var(logout, ml_keyname)

        # optimizer = AMSGrad.AMSGrad(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if OPTIMIZER == 'AMSGrad':
                ae_train_op = AMSGrad.AMSGrad(learning_rate).minimize(gen_loss, var_list = ae_varlist)
                metric_train_op = AMSGrad.AMSGrad(learning_rate).minimize(hinge_loss, global_step = global_step, var_list = ml_varlist)
            elif OPTIMIZER == 'adam':
                ae_train_op = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list = ae_varlist)
                metric_train_op = tf.train.AdamOptimizer(learning_rate).minimize(hinge_loss, global_step = global_step, var_list = ml_varlist)
            elif OPTIMIZER == 'RMSProp':
                ae_train_op = tf.train.RMSPropOptimizer(learning_rate, momentum = MOMENTUM).minimize(gen_loss, var_list = ae_varlist)
                metric_train_op = tf.train.RMSPropOptimizer(learning_rate, momentum = MOMENTUM).minimize(hinge_loss, global_step = global_step, var_list = ml_varlist)
            elif OPTIMIZER == 'Momentum':
                ae_train_op = tf.train.MomentumOptimizer(learning_rate, momentum = MOMENTUM, use_nesterov = True).minimize(gen_loss, var_list = ae_varlist)
                metric_train_op = tf.train.MomentumOptimizer(learning_rate, momentum = MOMENTUM, use_nesterov = True).minimize(hinge_loss, global_step = global_step, var_list = ml_varlist)
            else:
                ae_train_op = AMSGrad.AMSGrad(learning_rate).minimize(gen_loss, var_list = ae_varlist)
                metric_train_op = AMSGrad.AMSGrad(learning_rate).minimize(hinge_loss, global_step = global_step, var_list = ml_varlist)

            self.ae_train_op = ae_train_op
            self.metric_train_op = metric_train_op

        # comment for reduce tensorboard file size
        # with tf.name_scope('image-visualization'):
        #     photo_visual = tf.concat([photo_pl, pgen_image], 1)
        #     uds_visual = tf.concat([uds_pl, ugen_image], 1)
        #     image_visual = tf.concat([photo_visual, uds_visual], 2)
        #     tf.summary.image('image_visual', image_visual, max_outputs = 2)
        with tf.name_scope('ae-visualization'):
            tf.summary.scalar('photogen_loss', photogen_loss)
            tf.summary.scalar('udsvgen_loss', udsvgen_loss)
        with tf.name_scope('siame-visualization'):
            tf.summary.scalar('lr', learning_rate)
            tf.summary.scalar('hinge_loss', hinge_loss)
            tf.summary.scalar('accuracy', metricacc)
        self.summary = tf.summary.merge_all()
