import os

import numpy as np
import tensorflow as tf

import cmu_corridor_dataset_parser as data_parser
import FCNDatsetReader as Reader

reader = Reader.FCNDatsetReader(data_parser.read_dataset("../dataset/cmu_corridor_dataset", "png"))
saver = tf.train.Saver()

with tf.Session() as sess:
    # set keep_prob to 1, then it will simply ignore drop_out layer
    saver.restore(sess, "./output/model.ckpt")
    graph = tf.get_default_graph()
    valid_images, valid_annotations = reader.get_random_batch(2)
    pred = sess.run("pred_up:0", feed_dict={"input_image":valid_images})
