from __future__ import print_function
import tensorflow as tf
import scipy.io
import numpy as np
import pdb
import time


BASE = "/home/suri/workspace/style-transfer"
LOGDIR = BASE + "/summary"
VGG19_SAVED_WEIGHT_FILE = BASE + "/imagenet-vgg-verydeep-19.mat"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
COLOR_CHANNEL = 3
NOISE_RATIO = 0.6
ALPHA = 100  # style loss emphasis
BETA = 5  # content loss emphasis
# The mean to subtract from the input to the VGG model. This is the mean that
# when the VGG was used to train. Minor changes to this will make a lot of
# difference to the performance of model.
VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    def __init__(self, saved_weight=VGG19_SAVED_WEIGHT_FILE):
        self.data_dict = np.load(saved_weight, encoding='latin1').item()
        # for key, value in data_dict.iteritems(): 
        #     print(key, end=" ")
        print("weights file loaded")
        

    def get_conv_layer(self, prev_layer, name_of_layer):
        with tf.variable_scope(name):
            filter = self.get_filter_from_saved_file(name)
            conv_bias = self.get_bias_from_saved_file(name)
            conv_output = tf.nn.bias_add(tf.nn.conv2d(
                prev_layer, filter, [1, 1, 1, 1], padding="SAME"), conv_bias)
            conv_relu = tf.nn.relu(conv_output)
            return conv_relu

    def get_bias_from_saved_file(self, name_of_layer):
        return tf.constant(self.data_dict[name_of_layer][1], name="biases")

    def get_filter_from_saved_file(self, name_of_layer):
        return tf.constant(self.data_dict[name_of_layer][0], name="filter")

    def avg_pool_layer(self, prev_layer, name_of_layer):
        return tf.nn.avg_pool(prev_layer,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name=name_of_layer)

    def max_pool_layer(self, prev_layer, name_of_layer):
        return tf.nn.max_pool(prev_layer,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name=name_of_layer)

    def get_fc_layer(self, prev_layer, name):
        with tf.variable_scope(name):
            shape = prev_layer.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(prev_layer, [-1, dim])
            weights = self.get_weights_from_saved_file(name)
            biases = self.get_bias_from_saved_file(name)
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_weights_from_saved_file(self, name_of_layer):
        return tf.constant(self.data_dict[name_of_layer][0], name="weights")

    def create(self, rgb):
        start_time = time.time()
        rgb_scaled = rgb * 255.0
        with tf.variable.scope("bgr"):
            red, green, blue = tf.split(axis = 3, num_or_size_splits = 3, value = rgb_scaled)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(axis = 3, values = [blue - VGG_MEAN[0],
                                       green - VGG_MEAN[1],
                                       red - VGG_MEAN[2]])
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        self.conv1_1 = self.get_conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.get_conv_layer(self.conv1_1, "conv1_2")
        self.max_pool1 = self.max_pool_layer(self.conv1_2, "max_pool1")

        self.conv2_1 = self.get_conv_layer(self.max_pool1, "conv2_1")
        self.conv2_2 = self.get_conv_layer(self.conv2_1, "conv2_2")
        self.max_pool2 = self.max_pool_layer(self.conv2_2, "max_pool2")

        self.conv3_1 = self.get_conv_layer(self.max_pool2, "conv3_1")
        self.conv3_2 = self.get_conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.get_conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.get_conv_layer(self.conv3_3, "conv3_4")
        self.max_pool3 = self.max_pool_layer(self.conv3_4, "max_pool3")

        self.conv4_1 = self.get_conv_layer(self.max_pool3, "conv4_1")
        self.conv4_2 = self.get_conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.get_conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.get_conv_layer(self.conv4_3, "conv4_4")
        self.max_pool4 = self.max_pool_layer(self.conv4_4, "max_pool4")

        self.conv5_1 = self.get_conv_layer(self.max_pool4, "conv5_1")
        self.conv5_2 = self.get_conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.get_conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.get_conv_layer(self.conv5_3, "conv5_4")
        self.max_pool5 = self.max_pool_layer(self.conv5_4, "max_pool5")

        self.fc6 = self.get_fc_layer(self.max_pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.get_fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.get_fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, "prob")

        self.data_dict = None

        print("build finished %ds", (time.time()-start_time))
