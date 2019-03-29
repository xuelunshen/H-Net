import os
import random
import tensorflow as tf
from sean_functools import image_augment_tf

class tfio:
    """
    parent class for tfrecord io
    """
    def __init__(self, batch_size_tr, batch_size_v, batch_size_te, data_aug, width, height, channel, train_data_path, valid_data_path, test_data_path,
                 capacity, num_threads = 1):
        self.batch_size_tr = batch_size_tr
        self.batch_size_v = batch_size_v
        self.batch_size_te = batch_size_te
        self.data_aug = data_aug
        self.width = width
        self.height = height
        self.channel = channel
        self.capacity = capacity
        self.num_threads = num_threads
        self.min_after_dequeue = self.batch_size_tr*2
        
        file_list = os.listdir(train_data_path)
        random.shuffle(file_list)
        for i in range(len(file_list)):
            file_list[i] = train_data_path + file_list[i]
        self.train_file = file_list
        self.train_num = len(self.train_file) * 1000
        
        file_list = os.listdir(valid_data_path)
        # random.shuffle(file_list)
        for i in range(len(file_list)):
            file_list[i] = valid_data_path + file_list[i]
        self.valid_file = file_list
        self.valid_num = len(self.valid_file) * 1000
        
        file_list = os.listdir(test_data_path)
        # random.shuffle(file_list)
        for i in range(len(file_list)):
            file_list[i] = test_data_path + file_list[i]
        self.test_file = file_list
        self.test_num = len(self.test_file) * 1000

    def get_train_data(self):
        photo, uds, label = self.read_and_decode(self.train_file, True)
        return photo, uds, label

    def get_valid_data(self):
        photo, uds, label = self.read_and_decode(self.valid_file, False)
        return photo, uds, label

    def get_test_data(self):
        photo, uds, label, = self.read_and_decode(self.test_file, False)
        return photo, uds, label

    def read_and_decode(self, file_list, shuffle):
        files = tf.train.match_filenames_once(file_list)
        filename_queue = tf.train.string_input_producer(files, shuffle = shuffle, capacity = len(file_list) + 1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features = {'label':tf.FixedLenFeature([], tf.int64),
                                                                           'photo':tf.FixedLenFeature([], tf.string),
                                                                           'uds':tf.FixedLenFeature([], tf.string), })
    
        photo = tf.decode_raw(features['photo'], tf.uint8)
        photo = tf.reshape(photo, [self.width, self.height, self.channel])
    
        uds = tf.decode_raw(features['uds'], tf.uint8)
        uds = tf.reshape(uds, [self.width, self.height, self.channel])
    
        lbl = tf.cast(features['label'], tf.int32)

        # photo = tf.image.per_image_standardization(photo)
        # uds = tf.image.per_image_standardization(uds)
        
        return photo, uds, lbl

    def get_onebatch_traindata(self, width, height):
        train_p, train_u, train_label = self.get_train_data()
    
        if self.data_aug:
            train_p = image_augment_tf(train_p)
            train_u = image_augment_tf(train_u)

        train_p = tf.to_float(train_p) * 1.0 / 255.0
        train_u = tf.to_float(train_u) * 1.0 / 255.0
    
        if self.width != width:
            train_p = tf.image.resize_images(images = train_p, size = [width, height])
            train_u = tf.image.resize_images(images = train_u, size = [width, height])
    
        train_p_batch, train_u_batch, train_label_batch = tf.train.shuffle_batch([train_p, train_u, train_label],
                                                                                 batch_size = self.batch_size_tr,
                                                                                 num_threads = self.num_threads,
                                                                                 capacity = self.capacity,
                                                                                 min_after_dequeue = self.min_after_dequeue)
    
        return train_p_batch, train_u_batch, train_label_batch

    def get_onebatch_validdata(self, width, height):
        valid_p, valid_u, valid_label = self.get_valid_data()
    
        valid_p = tf.to_float(valid_p) * 1.0 / 255.0
        valid_u = tf.to_float(valid_u) * 1.0 / 255.0
    
        if self.width != width:
            valid_p = tf.image.resize_images(images = valid_p, size = [width, height])
            valid_u = tf.image.resize_images(images = valid_u, size = [width, height])
    
        valid_p_batch, valid_u_batch, valid_label_batch = tf.train.batch([valid_p, valid_u, valid_label],
                                                                         batch_size = self.batch_size_v,
                                                                         num_threads = self.num_threads,
                                                                         capacity = self.capacity)
    
        return valid_p_batch, valid_u_batch, valid_label_batch

    def get_onebatch_testdata(self, width, height):
        test_p, test_u, test_label = self.get_test_data()
    
        test_p = tf.to_float(test_p) * 1.0 / 255.0
        test_u = tf.to_float(test_u) * 1.0 / 255.0
    
        if self.width != width:
            test_p = tf.image.resize_images(images = test_p, size = [width, height])
            test_u = tf.image.resize_images(images = test_u, size = [width, height])
    
        test_p_batch, test_u_batch, test_label_batch = tf.train.batch([test_p, test_u, test_label],
                                                                      batch_size = self.batch_size_te,
                                                                      num_threads = self.num_threads,
                                                                      capacity = self.capacity)
    
        return test_p_batch, test_u_batch, test_label_batch
    
    def get_train_itr(self):
        return self.train_num // self.batch_size_tr
    
    def get_valid_itr(self):
        return self.valid_num // self.batch_size_v
    
    def get_test_itr(self):
        return self.test_num // self.batch_size_te
    
    def get_train_name(self):
        return self.train_file
    
    def get_valid_name(self):
        return self.valid_file
    
    def get_test_name(self):
        return self.test_file
