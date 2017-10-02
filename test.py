import sys
import argparse

import Image
import scipy
import scipy.misc
import numpy as np
import tensorflow as tf

import cmu_corridor_dataset_parser as data_parser
import FCNDatsetReader as Reader

FLAGS = None

def cvt_result2color(image, num_classes=20):
    """
    Convert given inferenced label image to color image
    """
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def main(_):
    reader = Reader.FCNDatsetReader(data_parser.read_dataset(FLAGS.data_dir, FLAGS.data_ext))
    with tf.Session() as sess:
        # set keep_prob to 1, then it will simply ignore drop_out layer
        saver = tf.train.import_meta_graph('./output/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./output/'))
        graph = tf.get_default_graph()
        valid_images, valid_annotations = reader.get_random_batch(2)
        pred = sess.run("pred_up:0", feed_dict={"input_image:0":valid_images,
                                                "keep_prob:0":1.0})
        color_image = cvt_result2color(pred[0], num_classes=2)
        print pred[0].shape
        print valid_annotations[0].shape
        scipy.misc.imsave('inference.png', color_image)
        im = Image.fromarray(np.squeeze(valid_annotations[0]) * 255)
        im.save('label.png')
        scipy.misc.imsave('original.png', valid_images[0])

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data_dir', type=str,
                        default='../dataset/cmu_corridor_dataset',
                        help='Directory of input dataset')
    PARSER.add_argument('--data_ext', type=str,
                        default='png',
                        help='File extension of image data')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
