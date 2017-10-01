"""
Fully convolutional network based on VGG-32 networks.
This code is written by modifying the codes from:
https://github.com/MarvinTeichmann/tensorflow-fcn
"""

from __future__ import print_function

import logging
from math import ceil
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68] # Mean values of B, G, and R.

class FCNVGG32(object):
    """
    Fully Convolutional Networks of VGG-32 model
    """
    def __init__(self, image, keep_prob, num_classes,
                 weight_decay=5e-4,
                 weights_path='DEFAULT'):
        """
        Build a tensor graph for VGG-32 FCNs.
        :param image: Placeholder for the input tensor.
        :param keep_prob: Placeholder of Dropout probability.
        :param num_classes: The number of classes in the dataset.
        :param weights_path: Complete path to the pretrained weight file. If the file is in the
        same folder with this code, just pass the filename with its extension. It should be either
        '*.npy' or '*.ckpt'. Pass empty string if you wish to train the network from scratch (not
        recommended).
        """
        # Parse input arguments into class variables
        self.image = image
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay

        if weights_path == 'DEFAULT':
            self.weights_path = 'vgg16.npy'
        else:
            self.weights_path = weights_path

        self.net_dict = np.load('vgg16.npy', encoding='latin1').item()

    def build(self, train=True):
        """
        Build a FCNs and assign variables from the preloaded pretrained file
        """
        # Preprocessing: convert RGB to BGR and subtract mean values of VGG images
        with tf.name_scope('Processing'):
            red, green, blue = tf.split(self.image, 3, 3)
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2]], axis=3)
            bgr = tf.Print(bgr, [tf.shape(bgr)],
                           message='Shape of input image: ',
                           summarize=4, first_n=1)

        # 1st Layer: Conv (w ReLU) -> Conv (w ReLU) -> Pool
        conv1_1 = self.__conv_layer(bgr, 'conv1_1')
        conv1_2 = self.__conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 2, 2, 2, 2, 'pool1')

        # 2nd Layer: Conv (w ReLU) -> Conv (w ReLU) -> Pool
        conv2_1 = self.__conv_layer(pool1, 'conv2_1')
        conv2_2 = self.__conv_layer(conv2_1, 'conv2_2')
        pool2 = self.max_pool(conv2_2, 2, 2, 2, 2, 'pool2')

        # 3rd Layer: Conv (w ReLU) -> Conv (w ReLU) -> Conv (w ReLU) -> Pool
        conv3_1 = self.__conv_layer(pool2, "conv3_1")
        conv3_2 = self.__conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.__conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 2, 2, 2, 2, 'pool3')

        # 4th Layer: Conv (w ReLU) -> Conv (w ReLU) -> Conv (w ReLU) -> Pool
        conv4_1 = self.__conv_layer(pool3, "conv4_1")
        conv4_2 = self.__conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.__conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 2, 2, 2, 2, 'pool4')

        # 5th Layer: Conv (w ReLU) -> Conv (w ReLU) -> Conv (w ReLU) -> Pool
        conv5_1 = self.__conv_layer(pool4, "conv5_1")
        conv5_2 = self.__conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.__conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 2, 2, 2, 2, 'pool5')

        # 6th Layer: replaced FC (Conv) layer
        # Converted from FC to Conv for FCN
        fc6 = self.__fc_layer(pool5, "fc6", new_shape=[7, 7, 512, 4096])
        if train:
            fc6 = tf.nn.dropout(fc6, self.keep_prob)

        # 7th Layer: replaced FC (Conv) layer
        # Converted from FC to Conv for FCN
        fc7 = self.__fc_layer(fc6, "fc7", new_shape=[1, 1, 4096, 4096])
        if train:
            fc7 = tf.nn.dropout(fc7, self.keep_prob)

        if train:
            score_fr = self.__score_layer(fc7, "score_fr")
        else:
            score_fr = self.__fc_layer(fc7, "fc8",
                                       new_shape=[1, 1, 4096, 1000],
                                       num_classes=self.num_classes,
                                       apply_relu=False)

        upscore = self.__deconv_layer(score_fr,
                                      shape=tf.shape(bgr),
                                      name='up',
                                      ksize=64, stride=32)

        pred_up = tf.argmax(upscore, axis=3, name="pred_up")

        return pred_up, upscore

    def __conv_layer(self, bottom, filt_name):
        with tf.variable_scope(filt_name):
            filt = self.__get_conv_filter(filt_name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.__get_bias(filt_name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            activation_summary(relu)
            return relu

    def __fc_layer(self, bottom, filt_name,
                   new_name=None,
                   new_shape=None,
                   num_classes=None,
                   apply_relu=True, debug=False):
        with tf.variable_scope(filt_name):
            if new_shape is not None:
                filt = self.__get_fc_weight_reshape(filt_name, new_shape, num_classes)

            if new_name is not None:
                filt_name = new_name

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.__get_bias(filt_name, num_classes=num_classes)
            activation = tf.nn.bias_add(conv, conv_biases)

            if apply_relu:
                activation = tf.nn.relu(activation)
            activation_summary(activation)

            if debug:
                activation = tf.Print(activation, [tf.shape(activation)],
                                      message='Shape of %s' % filt_name,
                                      summarize=4, first_n=1)
            return activation

    def __score_layer(self, bottom, name):
        with tf.variable_scope(name):
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, self.num_classes]
            # He initialization Sheme
            num_input = in_features
            stddev = (2 / num_input)**0.5
            # Apply convolution
            weights = self.__variable_with_weight_decay(shape, stddev, self.weight_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self.__bias_variable([self.num_classes], constant=0.0)
            activation = tf.nn.bias_add(conv, conv_biases)

            print('Layer ' + name + ': ' + str(weights.shape))
            activation_summary(activation)

            return activation

    def __deconv_layer(self, bottom, shape,
                       name, debug=True,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                height = ((in_shape[1] - 1) * stride) + 1
                width = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], height, width, self.num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], self.num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, self.num_classes, in_features]

            # create
            weights = self.__get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                print('Layer ' + name + ': ' + str(f_shape))
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        activation_summary(deconv)
        return deconv

    def __get_conv_filter(self, filt_name):
        """
        Return filter of given name from the pretrained file
        """
        init_weights = tf.constant_initializer(value=self.net_dict[filt_name][0],
                                               dtype=tf.float32)
        shape = self.net_dict[filt_name][0].shape
        print('Layer ' + filt_name + ': ' + str(shape))
        weights = tf.get_variable(name="filter", initializer=init_weights, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.weight_decay,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return weights

    def __get_deconv_filter(self, f_shape):
        """
        Compute bilinear filter and return it
        """
        filt_width = f_shape[0]
        filt_height = f_shape[1]
        half_width = ceil(filt_width/2.0)
        center = (2 * half_width - 1 - half_width % 2) / (2.0 * half_width)
        bilinear = np.zeros([filt_width, filt_height])
        for x in range(filt_width):
            for y in range(filt_height):
                value = (1 - abs(x / half_width - center)) * (1 - abs(y / half_width - center))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def __get_bias(self, name, num_classes=None):
        """
        Return bias of given name from the pretrained file
        """
        bias_wights = self.net_dict[name][1]
        shape = self.net_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self.__bias_reshape(bias_wights, shape[0],
                                              num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        return tf.get_variable(name="biases", initializer=init, shape=shape)

    def __bias_reshape(self, bweight, num_orig, num_new):
        """
        Build bias for filter produces with `__summary_reshape`
        """
        n_averaged_elements = num_orig // num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx // n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def __get_fc_weight_reshape(self, filt_name, shape, num_classes=None):
        print('Layer ' + filt_name + ': ' + str(shape))
        weights = self.net_dict[filt_name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self.__summary_reshape(weights, shape,
                                             num_new=num_classes)
        init_weights = tf.constant_initializer(value=weights,
                                               dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init_weights, shape=shape)

    def __variable_with_weight_decay(self, shape, stddev, wd_input):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd_input and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd_input, name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

    def __bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name='biases', shape=shape,
                               initializer=initializer)

    def max_pool(self, bottom, filter_height, filter_width, stride_y, stride_x, name,
                 padding='SAME'):
        """ Create a max pooling layer. """
        return tf.nn.max_pool(bottom, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def __summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
        :param fweight: original weights
        :param shape: shape of the desired fully-convolutional layer
        :param num_new: number of new classes

        Returns: Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert num_new < num_orig
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

def activation_summary(tensor):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    :param tensor: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = tensor.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', tensor)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))
