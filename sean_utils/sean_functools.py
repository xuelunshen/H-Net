import os
# import random
import argparse
# import functools
import numpy as np
from PIL import Image
import tensorflow as tf

def to_onehot(label):
    """
    change label from 1 to 00000001
    :param label: label
    :return: one hot label
    """
    label = (label + 1) / 2  #[1, -1] to [1, 0]
    label = np.reshape(label, [len(label), -1])
    nblabel = 1 - label
    label = np.concatenate((label, nblabel), axis = 1)
    
    return label

def str2bool(v):
    """
    function used in parser
    :param v: parameter
    :return: parameter
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ImageToMatrix(filename, width, height, channel):
    """
    change PIL.Image to matrix
    :param filename: file name
    :param width: width
    :param height: height
    :param channel: channel
    :return: image matrix
    """
    im = Image.open(filename)
    im = im.resize((width, height))
    # width, height = im.size
    im = list(im.getdata())
    new_data = np.reshape(im, (width, height, channel))
    new_data = new_data * 1.0 / 255.0
    return new_data

def sess_config(gpu_pp):
    """
    sesstion config
    :param gpu_pp: gpu propotion
    :return: config
    """
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = gpu_pp
    config.gpu_options.allow_growth = True
    return config

def get_var(logout, keyname):
    """
    get all variable in network
    :param logout: a function used to log information
    :param keyname: key name in network
    :return: var dic contrain BN parameters and bias || weight dic only contain weight
    """
    t_vars = tf.trainable_variables()
    logout('\t[*] printing all variables')
    log_variable(logout, t_vars)
    var_dic = {}
    weight_dic = {}
    for k in keyname:
        v = keyname[k]
        var = [var for var in t_vars if v in var.name]
        weight = [var for var in var if 'weights' in var.name and 'biases' not in var.name]
        var_dic[k] = var
        weight_dic[k] = weight
        logout('\t[*] printing ' + v + ' variables')
        log_variable(logout, weight)
    return var_dic, weight_dic

def log_variable(logout, var_list):
    """
    print variables
    :param logout: a function used to log information
    :param var_list: variable list
    :return: nothing but information in txt file
    """
    for idx, v in enumerate(var_list):
        logout('  var {:3}: {:15}   {}'.format(idx, str(v.get_shape()), v.name))

def get_learning_rate(global_step, one_epoch_step, lr, decay_step, decay_rate, lr_baseline):
    """
    get learning rate start at lr and decay DECAY RATE in each DACAY STEP and do not less than 0.0003
    :param global_step:
    :param one_epoch_step:
    :param lr:
    :param decay_step:
    :param decay_rate:
    :param lr_baseline:
    :return:
    """
    learning_rate = tf.train.exponential_decay(lr,  # Base learning rate.
                                               global_step,  # Current index into the dataset.
                                               decay_step * one_epoch_step,  # Decay step.
                                               decay_rate,  # Decay rate.
                                               staircase = True)
    learning_rate = tf.maximum(learning_rate, lr_baseline)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_accuracy_3000_smem(pfz, ufz, TOP, METRIC):
    T, F = 0, 0
    for i in range(np.shape(pfz)[0]):
        p = pfz[i]
        dt = np.squeeze(np_metric_distance(ufz, p, METRIC))
        
        if TOP == 1:
            idx = np.argmin(dt)
            if idx == i: T = T + 1
            else: F = F + 1
        elif TOP > 1:
            idx_order = np.argsort(dt)
            for j in range(TOP):
                idx = idx_order[j]
                if idx == i:
                    T = T + 1
                    break
                elif j + 1 == TOP:
                    F = F + 1
    accuracy = T * 1.0 / (T + F)
    return T, F, accuracy

def eval_accuracy(pfz, ufz, PNAME_LIST, UNAME_LIST, TOP, METRIC):
    """
    abandon function and please use eval_accuracy_advance except you wannna high time expensive
    evaluate network accuracy in dataset like HAND*
    :param pfz: np.array about photo feature vector
    :param ufz: bp.array about uds feature vector
    :param PNAME_LIST: photo name list
    :param UNAME_LIST: uds name list
    :param TOP: search for top *
    :param METRIC: metric
    :return: accuracy
    """
    T, F = 0, 0
    for i in range(len(PNAME_LIST)):
        PHOTO_NAME = PNAME_LIST[i]
        
        # srcfm = pfz[i]
        # ud_distance = np.sum(np.power(np.subtract(srcfm, ufz), 2), axis = 1)
        # ud_distance = np_metric_distance(srcfm, ufz, METRIC)
        # distance = np.reshape(ud_distance, (-1))
        distance = np.squeeze(np_metric_distance(ufz, pfz[i], METRIC))
        
        if TOP == 1:
            idx = np.argmin(distance)
            if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]: T = T + 1
            else: F = F + 1
        
        elif TOP > 1:
            idx_order = np.argsort(distance)
            for j in range(TOP):
                idx = idx_order[j]
                if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]:
                    T = T + 1
                    break
                else:
                    if j + 1 == TOP: F = F + 1
    acc = T * 1.0 / (T + F)
    # print('\n\tT:{:d} F:{:d} ACC:{:f}'.format(T, F, acc))
    return T, F, acc

def eval_accuracy_with_image(save_path, ppath, upath, pfz, ufz, PNAME_LIST, UNAME_LIST, TOP, METRIC):
    T, F = 0, 0
    timg = ImageToMatrix(ppath + '../../t.jpg', 256, 256, 3)
    fimg = ImageToMatrix(ppath + '../../f.jpg', 256, 256, 3)
    for i in range(len(PNAME_LIST)):
        PHOTO_NAME = PNAME_LIST[i]
        distance = np.squeeze(np_metric_distance(ufz, pfz[i], METRIC))
        img = ImageToMatrix(ppath + PHOTO_NAME, 256, 256, 3)
        gt = ImageToMatrix(upath + PHOTO_NAME.replace('I', 'U'), 256, 256, 3)
        row = np.concatenate((img, gt), axis = 1)
        finded = False
        
        if TOP == 1:
            idx = np.argmin(distance)
            prd = ImageToMatrix(upath + UNAME_LIST[idx], 256, 256, 3)
            row = np.concatenate((row, prd), axis = 1)
            if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]:
                T = T + 1
                row = np.concatenate((row, timg), axis = 1)
                finded = True
            else:
                F = F + 1
                row = np.concatenate((row, fimg), axis = 1)
                
        elif TOP > 1:
            idx_order = np.argsort(distance)
            for j in range(TOP):
                idx = idx_order[j]
                prd = ImageToMatrix(upath + UNAME_LIST[idx], 256, 256, 3)
                row = np.concatenate((row, prd), axis = 1)
                if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]:
                    finded = True

            if finded:
                T = T + 1
                row = np.concatenate((row, timg), axis = 1)
            else:
                F = F + 1
                row = np.concatenate((row, fimg), axis = 1)

        if finded: Image.fromarray((row * 255).astype(np.uint8)).save(save_path + '/' + str(i) + '_t.jpg')
        else: Image.fromarray((row * 255).astype(np.uint8)).save(save_path + '/' + str(i) + '_f.jpg')
        
    acc = T * 1.0 / (T + F)
    return T, F, acc

def eval_accuracy_advance(pfz, ufz, PNAME_LIST, UNAME_LIST, TOP, METRIC):
    """
    faster than eval_accuracy with matrix calculate
    evaluate network accuracy in dataset like HAND*
    :param pfz: np.array about photo feature vector
    :param ufz: bp.array about uds feature vector
    :param PNAME_LIST: photo name list
    :param UNAME_LIST: uds name list
    :param TOP: search for top *
    :param METRIC: metric
    :return: accuracy
    """
    T, F = 0, 0
    # p = np.expand_dims(pfz, axis = 1)
    # udmat = np.sum(np.power(np.subtract(p, ufz), 2), axis = 2)
    udmat = np.squeeze(np_metric_distance(np.expand_dims(pfz, axis = 1), ufz, METRIC))
    
    if TOP == 1:
        idxmat = np.argmin(udmat, axis = 1)
    else:
        idxmat = np.argsort(udmat, axis = 1)
    
    for i in range(len(PNAME_LIST)):
        PHOTO_NAME = PNAME_LIST[i]
        
        if TOP == 1:
            idx = idxmat[i]
            if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]: T = T + 1
            else: F = F + 1
            
        elif TOP > 1:
            idx_order = idxmat[i]
            for j in range(TOP):
                idx = idx_order[j]
                if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]:
                    T = T + 1
                    break
                else:
                    if j + 1 == TOP: F = F + 1
                    
    acc = T * 1.0 / (T + F)
    # print('\n\tT:{:d} F:{:d} ACC:{:f}'.format(T, F, acc))
    return T, F, acc

def eval_accuracy_advance_with_image(ppath, upath, pfz, ufz, PNAME_LIST, UNAME_LIST, TOP, METRIC):
    """
    faster than eval_accuracy with matrix calculate
    evaluate network accuracy in dataset like HAND*
    :param ppath: path to save photo img
    :param upath: path to save uds img
    :param pfz: np.array about photo feature vector
    :param ufz: bp.array about uds feature vector
    :param PNAME_LIST: photo name list
    :param UNAME_LIST: uds name list
    :param TOP: search for top *
    :param METRIC: metric
    :return: accuracy
    """
    T, F = 0, 0
    # p = np.expand_dims(pfz, axis = 1)
    # udmat = np.sum(np.power(np.subtract(p, ufz), 2), axis = 2)
    udmat = np.squeeze(np_metric_distance(np.expand_dims(pfz, axis = 1), ufz, METRIC))

    if TOP == 1:
        idxmat = np.argmin(udmat, axis = 1)
    else:
        idxmat = np.argsort(udmat, axis = 1)
    
    timg = ImageToMatrix(ppath + '../../t.jpg', 256, 256, 3)
    fimg = ImageToMatrix(ppath + '../../f.jpg', 256, 256, 3)
    result = None
    for i in range(len(PNAME_LIST)):
        PHOTO_NAME = PNAME_LIST[i]
        img = ImageToMatrix(ppath + PHOTO_NAME, 256, 256, 3)
        gt = ImageToMatrix(upath + PHOTO_NAME.replace('I', 'U'), 256, 256, 3)
        row = np.concatenate((img, gt), axis = 1)
        finded = False
        
        if TOP == 1:
            idx = idxmat[i]
            prd = ImageToMatrix(upath + UNAME_LIST[idx], 256, 256, 3)
            row = np.concatenate((row, prd), axis = 1)
            if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]:
                T = T + 1
                row = np.concatenate((row, timg), axis = 1)
            else:
                F = F + 1
                row = np.concatenate((row, fimg), axis = 1)
        
        elif TOP > 1:
            idx_order = idxmat[i]
            for j in range(TOP):
                idx = idx_order[j]
                prd = ImageToMatrix(upath + UNAME_LIST[idx], 256, 256, 3)
                row = np.concatenate((row, prd), axis = 1)
                if PHOTO_NAME.replace('I', 'U') == UNAME_LIST[idx]:
                    finded = True
            
            if finded:
                T = T + 1
                row = np.concatenate((row, timg), axis = 1)
            else:
                F = F + 1
                row = np.concatenate((row, fimg), axis = 1)
        
        if result is not None: result = np.concatenate((result, row), axis = 0)
        else: result = row
    
    acc = T * 1.0 / (T + F)
    return T, F, acc, result

def genloss(img1, img2):
    """
    get generation loss
    :param img1: image 1
    :param img2: image 2
    :return: generation loss
    """
    """calc image generation loss"""
    batch_size = tf.shape(img1)[0]
    img1_flat = tf.reshape(img1, [batch_size, -1])
    img2_flat = tf.reshape(img2, [batch_size, -1])
    BCE = tf.reduce_sum(tf.square(img1_flat - img2_flat), reduction_indices = 1)
    return tf.reduce_mean(BCE)

def top_search(pfz, ufz, PNAME_LIST, UNAME_LIST, PHOTO_LIST, UDS_LIST, TOP, File_ID, METRIC):
    """
    top search in unlabel dataset not HAND
    :param pfz: photo feature vector
    :param ufz: uds feature vector
    :param PNAME_LIST: photo name list
    :param UNAME_LIST: uds name list
    :param PHOTO_LIST: photo list
    :param UDS_LIST: uds list
    :param TOP: top *
    :param File_ID: file ID
    :param METRIC: metric
    :return: a file contain search result
    """
    PREDLOGDIR = 'Search-' + File_ID + '-TOP' + str(TOP)
    if not os.path.exists(PREDLOGDIR):
        os.mkdir(PREDLOGDIR)
    
    # p = np.expand_dims(pfz, axis = 1)
    # udmat = np.sum(np.power(np.subtract(p, ufz), 2), axis = 2)
    # udmat = np.sum(np.power(np.subtract(p, ufz), 2), axis = 2)
    # udmat = np_metric_distance(p, ufz, METRIC)
    udmat = np.squeeze(np_metric_distance(np.expand_dims(pfz, axis = 1), ufz, METRIC))

    if TOP == 1:
        idxmat = np.argmin(udmat, axis = 1)
    else:
        idxmat = np.argsort(udmat, axis = 1)

    
    num = len(PNAME_LIST)
    for i in range(num):
        print('{:d}/{:d}'.format(i, num))
        
        ONE_PHOTO = PHOTO_LIST[i]
        PHOTO_NAME = PNAME_LIST[i]
        photo_fv = pfz[i, :]
        
        if TOP == 1:
            idx = idxmat[i]
            
            ONE_UDS = UDS_LIST[idx]
            uds_fv = ufz[idx]
            ud = udmat[i][idx]
            UDS_NAME = UNAME_LIST[idx]
            
            p = Image.fromarray((ONE_PHOTO * 255).astype(np.uint8))
            p.save('%s/%.5f-p-%s-%s' % (PREDLOGDIR, ud, PHOTO_NAME, UDS_NAME))
            
            u = Image.fromarray((ONE_UDS * 255).astype(np.uint8))
            u.save('%s/%.5f-u-%s-%s' % (PREDLOGDIR, ud, PHOTO_NAME, UDS_NAME))
            
            PREDLOG_FOUT = open(str('%s/%.5f-%s-%s.txt' % (PREDLOGDIR, ud, PHOTO_NAME, UDS_NAME)), 'w')
            PREDLOG_FOUT.write('{:s} {:s} {:.5f}\n\n'.format(PHOTO_NAME, UDS_NAME, ud))
            
            PREDLOG_FOUT.write('photo feature vector:\n')
            for _ in photo_fv:
                PREDLOG_FOUT.write(str(_) + ' ')
            PREDLOG_FOUT.write('\n\n')
            
            PREDLOG_FOUT.write('uds feature vector:\n')
            for _ in uds_fv:
                PREDLOG_FOUT.write(str(_) + ' ')
            PREDLOG_FOUT.write('\n\n')
        
        elif TOP > 1:
            idx_order = idxmat[i]

            p = Image.fromarray((ONE_PHOTO * 255).astype(np.uint8))
            p.save('%s/%d-p-%s' % (PREDLOGDIR, i, PHOTO_NAME))
            PREDLOG_FOUT = open((PREDLOGDIR + '/' + str(i) + "-" + PHOTO_NAME + '.txt'), 'w')
            PREDLOG_FOUT.write('photo feature vector:\n')
            
            for _ in photo_fv:
                PREDLOG_FOUT.write(str(_) + ' ')
            PREDLOG_FOUT.write('\n\n')
            
            for j in range(TOP):
                idx = idx_order[j]
                uds_fv = ufz[idx]
                ud = udmat[i][idx]
                UDS_NAME = UNAME_LIST[idx]

                u = Image.fromarray((UDS_LIST[idx] * 255).astype(np.uint8))
                u.save('%s/%d-%.3f-%s' % (PREDLOGDIR, i, ud, UDS_NAME))
                
                PREDLOG_FOUT.write('{:s} {:s} {:.5f}\n'.format(PHOTO_NAME, UDS_NAME, ud))
                PREDLOG_FOUT.write('uds feature vector:\n')
                
                for _ in uds_fv:
                    PREDLOG_FOUT.write(str(_) + ' ')
                PREDLOG_FOUT.write('\n')
                PREDLOG_FOUT.write('\n')

def visual_feature(pfz, ufz, PNAME_LIST, UNAME_LIST, File_ID):
    """
    visualiza feature vector
    :param pfz: photo feature vector
    :param ufz: uds feature vector
    :param PNAME_LIST: photo name list
    :param UNAME_LIST: uds name list
    :param File_ID: file ID
    :return:
    """
    CHARLOGDIR = 'VisualFeature-' + File_ID
    if not os.path.exists(CHARLOGDIR):
        os.mkdir(CHARLOGDIR)
    
    PCHARLOG_FOUT = open(str('%s/photo_fv.txt' % CHARLOGDIR), 'w')
    PNAMELOG_FOUT = open(str('%s/photo_name.txt' % CHARLOGDIR), 'w')
    for i in range(len(PNAME_LIST)):
        print('p : {:d}/{:d}'.format(i, len(PNAME_LIST)))
        
        PHOTO_NAME = PNAME_LIST[i]
        pfv = pfz[i, :]
        
        PNAMELOG_FOUT.write(PHOTO_NAME + '\n')
        for _ in pfv:
            PCHARLOG_FOUT.write(str(_) + ' ')
        PCHARLOG_FOUT.write('\n')
    
    UCHARLOG_FOUT = open(str('%s/uds_fv.txt' % CHARLOGDIR), 'w')
    UNAMELOG_FOUT = open(str('%s/uds_name.txt' % CHARLOGDIR), 'w')
    for j in range(len(UNAME_LIST)):
        print('u : {:d}/{:d}'.format(j, len(UNAME_LIST)))
        
        UDS_NAME = UNAME_LIST[j]
        ufv = ufz[j, :]
        
        UNAMELOG_FOUT.write(UDS_NAME + '\n')
        for _ in ufv:
            UCHARLOG_FOUT.write(str(_) + ' ')
        UCHARLOG_FOUT.write('\n')

def image_augment_tf(image):
    def bri(): return tf.image.random_brightness(image, max_delta = 0.2)
    def f1():return image
    image = tf.cond(tf.random_uniform([]) > 0.5, bri, f1)

    # def sat(): return tf.image.random_saturation(image, lower = 0.3, upper = 1.8)
    # def f2():return image
    # image = tf.cond(tf.random_uniform([]) > 0.5, sat, f2)

    # def hue(): return tf.image.random_hue(image, max_delta = 0.3)
    # def f3():return image
    # image = tf.cond(tf.random_uniform([]) > 0.5, hue, f3)

    def con(): return tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
    def f4():return image
    image = tf.cond(tf.random_uniform([]) > 0.5, con, f4)

    # def gas(): return tf.add(image, tf.cast(tf.random_normal(shape = image.get_shape(), mean = 0.0, stddev = 20, dtype = tf.float32), tf.uint8))
    # def f5():return image
    # image = tf.cond(tf.random_uniform([]) > 0.5, gas, f5)
    return image

def saveLargeImage(img, gap, filename):
    dims = np.shape(img)[0]
    if dims > gap:
        if dims % gap == 0:
            n = dims // gap
        else:
            n = dims // gap + 1
        for i in range(n):
            if (i+1)*gap >= dims:
                cut = img[i * gap:dims]
            else:
                cut = img[i*gap:(i+1)*gap]
            Image.fromarray((cut * 255).astype(np.uint8)).save(filename + '-' + str(i) + '.jpg')
    else:
        Image.fromarray((img * 255).astype(np.uint8)).save(filename + '.jpg')

def feature_triplet_loss(pz_mean, uz_mean, margin, METRIC):
    p = tf.expand_dims(pz_mean, axis = 1)
    u = uz_mean
    
    # get a euclidean distance matrix size of BN*BN*VSIZE
    # l = tf.subtract(p, u)
    # l2 = tf.pow(l, 2)
    
    # matrix size of BN*BN
    # the the diagonal value is correspond photo and uds euclidean distance, it should be minimum
    # D_all = tf.reduce_sum(l2, axis = 2) # the l3 contain D(a,p) and D(a,n) and D(a,p) position is diag_part
    D_all = tf.squeeze(tf_metric_distance(p, u, METRIC))
    
    # save the euclidean distance of corresponed photo and uds
    D_ap = tf.diag_part(D_all)
    D_ap_m = tf.expand_dims(D_ap + margin, axis = 1)
    
    triplet_loss = tf.subtract(D_ap_m, D_all)
    # triplet_loss = tf.maximum(0.0, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # nonzeros_1 = tf.reduce_sum(tf.to_float(tf.greater(triplet_loss, 1e-16)))
    # nonzeros_2 = tf.reduce_sum(tf.to_float(tf.greater(tf.diag_part(triplet_loss), 1e-16)))
    nonzeros_1 = tf.reduce_sum(tf.to_float(tf.greater(triplet_loss, 0.0)))
    nonzeros_2 = tf.reduce_sum(tf.to_float(tf.greater(tf.diag_part(triplet_loss), 0.0)))
    nonzeros = nonzeros_1 - nonzeros_2
    
    loss_sum = tf.reduce_sum(triplet_loss) - tf.reduce_sum(tf.diag_part(triplet_loss))
    # loss_mean = loss_sum / (nonzeros + 1e-16)
    # loss_mean = loss_sum / nonzeros
    def f1(): return tf.to_float(0)
    def f2():return loss_sum / nonzeros
    loss_mean = tf.cond(tf.equal(nonzeros,0), f1, f2)
    
    return loss_mean

def tf_square_euclidean_distance(pz_mean, uz_mean):
    """
    input a BN*VSIZE and BN*VSIZE feature vector
    return a BN*1 square euclidean distance

    :param pz_mean: BN*VSIZE feature vector
    :param uz_mean: BN*VSIZE feature vector
    :return: BN*1 square euclidean distance
    """
    euclidean_distance = tf.reduce_sum(tf.square(tf.subtract(pz_mean, uz_mean)), axis = tf.rank(pz_mean) - 1, keepdims = True)
    return euclidean_distance

def tf_standard_euclidean_distance(pz_mean, uz_mean):
    """
    input a BN*VSIZE and BN*VSIZE feature vector
    return a BN*1 square euclidean distance
    :param pz_mean: BN*VSIZE feature vector
    :param uz_mean: BN*VSIZE feature vector
    :return: BN*1 standard euclidean distance
    """
    standard_euclidean_distance = tf.sqrt(tf_square_euclidean_distance(pz_mean, uz_mean))
    return standard_euclidean_distance

def tf_cosin_distance(pz_mean, uz_mean):
    """
    calculate for cosin similarity with tfrecord
    :param pz_mean: feature vector
    :param uz_mean: feature vector
    :return: cosin similarity
    """
    p_norm = tf.sqrt(tf.reduce_sum(tf.square(pz_mean), axis = tf.rank(pz_mean) - 1))
    u_norm = tf.sqrt(tf.reduce_sum(tf.square(uz_mean), axis = tf.rank(uz_mean) - 1))
    p_inproduct_u = tf.reduce_sum(tf.multiply(pz_mean, uz_mean), axis = tf.rank(pz_mean) - 1)
    cosine = tf.divide(p_inproduct_u, tf.multiply(p_norm, u_norm))
    cosine = -cosine + 1 # change [-1, 1] decrase to [0, 2] increase
    return cosine

def tf_metric_distance(pz_mean, uz_mean, metric):
    if 'ED' == metric:
        return tf_standard_euclidean_distance(pz_mean, uz_mean)
    elif 'COSINE' == metric:
        return tf_cosin_distance(pz_mean, uz_mean)
    elif 'SED' == metric:
        return tf_square_euclidean_distance(pz_mean, uz_mean)
    
    return tf_square_euclidean_distance(pz_mean, uz_mean)

def np_square_euclidean_distance(pz_mean, uz_mean):
    euclidean_distance = np.sum(np.square(np.subtract(pz_mean, uz_mean)), axis = np.ndim(pz_mean) - 1, keepdims = True)
    return euclidean_distance

def np_standard_euclidean_distance(pz_mean, uz_mean):
    standard_euclidean_distance = np.sqrt(np_square_euclidean_distance(pz_mean, uz_mean))
    return standard_euclidean_distance

def np_cosin_distance(pz_mean, uz_mean):
    p_norm = np.sqrt(np.sum(np.square(pz_mean), axis = np.ndim(pz_mean) - 1))
    u_norm = np.sqrt(np.sum(np.square(uz_mean), axis = np.ndim(uz_mean) - 1))
    p_inproduct_u = np.sum(np.multiply(pz_mean, uz_mean), axis = np.ndim(pz_mean) - 1)
    cosine = np.divide(p_inproduct_u, np.multiply(p_norm, u_norm))
    cosine = -cosine + 1 # change [-1, 1] decrase to [0, 2] increase
    return cosine

def np_metric_distance(pz_mean, uz_mean, metric):
    if 'ED' == metric:
        return np_standard_euclidean_distance(pz_mean, uz_mean)
    elif 'COSINE' == metric:
        return np_cosin_distance(pz_mean, uz_mean)
    elif 'SED' == metric:
        return np_square_euclidean_distance(pz_mean, uz_mean)
    
    return np_square_euclidean_distance(pz_mean, uz_mean)

########################################################################################################################
def get_metricacc(pred, label_pl):
    predict = tf.cast((pred > 0), tf.int32) * 2 - 1
    correct_prediction = tf.cast(tf.equal(predict, label_pl), tf.float64)
    metricacc = tf.reduce_mean(correct_prediction)
    return metricacc, predict

def hingeloss(label_pl, pred):
    temp = 1 - tf.multiply(tf.cast(label_pl, tf.float32), pred)
    hinge_loss = tf.reduce_sum(tf.maximum(0.0, temp))
    return hinge_loss