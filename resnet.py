from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from ops import *
from data import DataSet
import time
from datetime import datetime
import os
import sys


class ResNet(object):

    def __init__(self, train=True, common_params=None, net_params=None):
        self.train = train
        self.weight_decay = 0.0
        if common_params:
            gpu_nums = len(str(common_params['gpus']).split(','))
            self.batch_size = int(int(common_params['batch_size']) / gpu_nums)
        if net_params:
            self.weight_decay = float(net_params['weight_decay'])

    def inference(self, data_l):
        conv_num = 1

        # conv1
        temp_shortcut = conv2d('conv_shortcut1', data_l,
                               [1, 1, 1, 64], stride=2)

        temp_conv = conv2d('conv{}'.format(conv_num), data_l,
                           [3, 3, 1, 64], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 64, 64], stride=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_1', temp_conv, train=self.train)

        temp_conv = temp_shortcut + temp_conv

        # conv2
        temp_shortcut = conv2d('conv_shortcut2', temp_conv,
                               [1, 1, 64, 128], stride=2)

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 64, 128], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 128, 128], stride=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_2', temp_conv, train=self.train)

        temp_conv = temp_shortcut + temp_conv

        # conv3
        temp_shortcut = conv2d('conv_shortcut3', temp_conv,
                               [1, 1, 128, 256], stride=2)

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 128, 256], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 256, 256], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 256, 256], stride=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_3', temp_conv, train=self.train)

        temp_conv = temp_shortcut + temp_conv

        # conv4
        temp_shortcut = conv2d('conv_shortcut4', temp_conv,
                               [1, 1, 256, 512], stride=1)
        # temp_shortcut = temp_conv

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 256, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_4', temp_conv, train=self.train)

        temp_conv = temp_shortcut + temp_conv

        # conv5
        # temp_shortcut = conv2d('conv_shortcut5', temp_conv,
        #                        [1, 1, 512, 512], stride=1)

        temp_shortcut = temp_conv
        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, dilation=2,
                           wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, dilation=2,
                           wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, dilation=2,
                           wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_5', temp_conv, train=self.train)

        temp_conv = temp_shortcut + temp_conv

        # conv6
        # temp_shortcut = conv2d('conv_shortcut6', temp_conv,
        #                        [1, 1, 512, 512], stride=1)
        temp_shortcut = temp_conv

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, dilation=2,
                           wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, dilation=2,
                           wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, dilation=2,
                           wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_6', temp_conv, train=self.train)

        temp_conv = temp_shortcut + temp_conv

        # conv7
        # temp_shortcut = conv2d('conv_shortcut7', temp_conv,
        #                        [1, 1, 512, 512], stride=1)
        temp_shortcut = temp_conv

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_7', temp_conv, train=self.train)

        temp_conv = temp_shortcut + temp_conv

        # conv8
        temp_shortcut = deconv2d('conv_shortcut8', temp_conv,
                               [1, 1, 512, 256], stride=2)

        temp_conv = deconv2d('conv{}'.format(conv_num), temp_conv,
                             [4, 4, 512, 256], stride=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 256, 256], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [3, 3, 256, 256], stride=1, wd=self.weight_decay)

        temp_conv = temp_shortcut + temp_conv
        conv_num += 1

        # Unary prediction
        temp_conv = conv2d('conv{}'.format(conv_num), temp_conv,
                           [1, 1, 256, 313], stride=1, relu=False,
                           wd=self.weight_decay)
        conv_num += 1

        conv8_313 = temp_conv

        return conv8_313

    def loss(self, scope, conv8_313, prior_boost_nongray, gt_ab_313):

        flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
        flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1, 313])

        g_loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=flat_conv8_313,
                labels=flat_gt_ab_313)) / (self.batch_size)

        tf.summary.scalar('weight_loss',
                          tf.add_n(tf.get_collection('losses', scope=scope)))
        #
        dl2c = tf.gradients(g_loss, conv8_313)
        dl2c = tf.stop_gradient(dl2c)
        #
        new_loss = tf.reduce_sum(dl2c * conv8_313 * prior_boost_nongray) + \
            tf.add_n(tf.get_collection('losses', scope=scope))

        return new_loss, g_loss
