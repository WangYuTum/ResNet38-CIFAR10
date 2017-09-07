#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import nn
import sys
import os
sys.path.append("..")
import data_utils as dt

class ResNet38:
    def __init__(self, weight_path=None):
        self._weight_dict = dt.load_weight(weight_path)
        self._var_dict = {}
        self._num_classes = 10
        self._poolsize = [1,3,3,1]
        self._poolstride = [1,2,2,1]

    def _build_model(self, image, is_train=False):
        '''If is_train, save weight to self._var_dict,
           otherwise, don't save weights'''

        model = {}
        feed_dict = self._weight_dict
        if is_train:
            var_dict = self._var_dict
        else:
            var_dict = None

        if is_train:
            dropout = True
        else:
            dropout = False

        # Do not use dropout for CIFAR10 classification
        dropout = False

        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        # B0: [H,W,3] -> [H,W,64]
        with tf.variable_scope('B0'):
            model['B0'] = nn.conv_layer(image, feed_dict, 1, 'SAME',
                                        shape_dict['B0'], var_dict)

        # Pool0: [H,W,64] -> [H/2, W/2, 64]
        with tf.variable_scope('Pool0'):
            model['Pool0'] = nn.max_pool(model['B0'], ksize=self._poolsize, stride=self._poolstride)
        # B2_1: [H/2, W/2, 64] -> [H/2, W/2, 128]
        # Note that downsampling is performed via max pooling, not the ResUnit function.
        # The name of the function is missleading because of historical reason
        shape_dict['B2'] = {}
        shape_dict['B2']['side'] = [1,1,64,128]
        shape_dict['B2']['convs'] = [[3,3,64,128],[3,3,128,128]]
        with tf.variable_scope('B2_1'):
            model['B2_1'] = nn.ResUnit_downsample_2convs(model['Pool0'],
                                                         feed_dict,
                                                         shape_dict['B2'],
                                                         var_dict=var_dict)
        # B2_2, B2_3: [H/2, W/2, 128]
        for i in range(2):
            with tf.variable_scope('B2_'+str(i+2)):
                model['B2_'+str(i+2)] = nn.ResUnit_2convs(model['B2_'+str(i+1)], feed_dict,
                                                          shape_dict['B2']['convs'][1],
                                                          var_dict=var_dict)
        # Pool1: [H/2, W/2, 128] -> [H/4, W/4, 128]
        with tf.variable_scope('Pool1'):
            model['Pool1'] = nn.max_pool(model['B2_3'], ksize=self._poolsize, stride=self._poolstride)
        # B3_1: [H/4, W/4, 128] -> [H/4, W/4, 256]
        shape_dict['B3'] = {}
        shape_dict['B3']['side'] = [1,1,128,256]
        shape_dict['B3']['convs'] = [[3,3,128,256],[3,3,256,256]]
        with tf.variable_scope('B3_1'):
            model['B3_1'] = nn.ResUnit_downsample_2convs(model['Pool1'],
                                                         feed_dict,
                                                         shape_dict['B3'],
                                                         var_dict=var_dict)
        # B3_2, B3_3: [H/4, W/4, 256]
        for i in range(2):
            with tf.variable_scope('B3_'+str(i+2)):
                model['B3_'+str(i+2)] = nn.ResUnit_2convs(model['B3_'+str(i+1)], feed_dict,
                                                          shape_dict['B3']['convs'][1],
                                                          var_dict=var_dict)
        # Pool2: [H/4, W/4, 256] -> [H/8, W/8, 256]
        with tf.variable_scope('Pool2'):
            model['Pool2'] = nn.max_pool(model['B3_3'], ksize=self._poolsize, stride=self._poolstride)
        # B4_1: [H/8, W/8, 256] -> [H/8, W/8, 512]
        shape_dict['B4'] = {}
        shape_dict['B4']['side'] = [1,1,256,512]
        shape_dict['B4']['convs'] = [[3,3,256,512],[3,3,512,512]]
        with tf.variable_scope('B4_1'):
            model['B4_1'] = nn.ResUnit_downsample_2convs(model['Pool2'],
                                                             feed_dict,
                                                             shape_dict['B4'],
                                                             var_dict=var_dict)
        # B4_2 ~ B4_6: [H/8, W/8, 512]
        for i in range(5):
            with tf.variable_scope('B4_'+str(i+2)):
                model['B4_'+str(i+2)] = nn.ResUnit_2convs(model['B4_'+str(i+1)],
                                                               feed_dict,
                                                               shape_dict['B4']['convs'][1],
                                                               var_dict=var_dict)
        # B5_1: [H/8, W/8, 512] -> [H/8, W/8, 1024]
        shape_dict['B5_1'] = {}
        shape_dict['B5_1']['side'] = [1,1,512,1024]
        shape_dict['B5_1']['convs'] = [[3,3,512,512],[3,3,512,1024]]
        with tf.variable_scope('B5_1'):
            model['B5_1'] = nn.ResUnit_downsample_2convs(model['B4_6'],
                                                               feed_dict,
                                                               shape_dict['B5_1'],
                                                               var_dict=var_dict)
        # B5_2, B5_3: [H/8, W/8, 1024]
        # Shape for B5_2, B5_3
        shape_dict['B5_2_3'] = [[3,3,1024,512],[3,3,512,1024]]
        for i in range(2):
            with tf.variable_scope('B5_'+str(i+2)):
                model['B5_'+str(i+2)] = nn.ResUnit_bottleneck_2convs(model['B5_'+str(i+1)],
                                                  feed_dict, shape_dict['B5_2_3'],
                                                  var_dict=var_dict)

        # B6: [H/8, W/8, 1024] -> [H/8, W/8, 2048]
        shape_dict['B6'] = [[1,1,1024,512],[3,3,512,1024],[1,1,1024,2048]]
        with tf.variable_scope('B6'):
            model['B6'] = nn.ResUnit_bottleneck_3conv(model['B5_3'],
                                                             feed_dict,
                                                             shape_dict['B6'],
                                                             dropout=dropout,
                                                             var_dict=var_dict)
        # B7: [H/8, W/8, 2048] -> [H/8, W/8, 4096]
        shape_dict['B7'] = [[1,1,2048,1024],[3,3,1024,2048],[1,1,2048,4096]]
        with tf.variable_scope('B7'):
            model['B7'] = nn.ResUnit_bottleneck_3conv(model['B6'],
                                                             feed_dict,
                                                             shape_dict['B7'],
                                                             dropout=dropout,
                                                             var_dict=var_dict)

        # ResNet tail. No conv, only batch_norm + activation
        shape_dict['Tail'] = 4096
        with tf.variable_scope('Tail'):
            model['Tail'] = nn.ResUnit_tail(model['B7'], feed_dict,
                                            shape_dict['Tail'], var_dict)

        # Global Pooling: [H/8, W/8, 4096] -> [4096]
        with tf.variable_scope('avg_pool'):
            model['pool_out'] = nn.Global_avg_pool(model['Tail'])

        # Fully connected: [4096] -> [10]
        with tf.variable_scope('FC'):
            batch_size = tf.shape(model['pool_out'])[0]
            model['fc_out'] = nn.FC(model['pool_out'], batch_size, feed_dict,
                                    self._num_classes, shape_dict['Tail'], var_dict)

        return model

    def _weight_decay(self, decay_rate):
        '''Compute weight decay loss for convolution kernel and fully connected
        weigths, excluding trainable variables of BN layer'''

        l2_losses = []
        for var in tf.trainable_variables():
            if var.op.name.find('kernel') or var.op.name.find('fc_weight') or var.op.name.find('fc_bias'):
                l2_losses.append(tf.nn.l2_loss(var))

        return tf.multiply(decay_rate, tf.add_n(l2_losses))

    def num_parameters(self):
        '''Compute the number of trainable parameters. Note that it MUST be called after the graph is built'''

        return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])

    def train(self, image, label, params):
        '''Input: Image [batch_size, H, W, C]
                  Label [batch_size]
                  params: 'num_class', 'batch_size', 'decay_rate' '''

        # Here padding to 36x36 then randomly crop per image.
        padded_img = tf.image.resize_image_with_crop_or_pad(image, 36, 36)
        cropped_img = tf.random_crop(padded_img, [params['batch_size'], 32, 32,3])
        # padding and cropping end

        # Here randomly flip each image
        flipped_img = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped_img)
        # randomly flipping end

        model = self._build_model(flipped_img, is_train=True)
        prediction = model['fc_out']
        label = tf.reshape(label, [params['batch_size']])

        # compute train accuracy
        pred_label = tf.argmax(tf.nn.softmax(prediction), axis=1)
        correct_pred_bools = tf.equal(pred_label, label)
        correct_preds = tf.reduce_sum(tf.cast(correct_pred_bools, tf.float32))
        train_acc = correct_preds/params['batch_size']
        entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,
                                                                      logits=prediction))
        total_loss = entropy_loss + self._weight_decay(params['decay_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Momentum lr=0.1, momentum=0.9
            train_op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(total_loss)

        return train_op, total_loss, train_acc, correct_preds

    def inf(self, image, label, params):
        '''Input: Image [batch_size, H, W, C]
                  Label [batch_size]
                  params: 'batch_size', 'feed_path' '''

        model = self._build_model(image, is_train=False)
        prediction = model['fc_out']
        label = tf.reshape(label, [params['batch_size']])

        pred_label = tf.argmax(tf.nn.softmax(prediction), axis=1)
        correct_pred_bools = tf.equal(pred_label, label)
        correct_preds = tf.reduce_sum(tf.cast(correct_pred_bools, tf.float32))

        return correct_preds

