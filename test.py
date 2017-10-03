import sys
import argparse

import Image
import scipy
import scipy.misc
import numpy as np
import tensorflow as tf

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
    """
    """
    if FLAGS.image is None:
        print '--image must be given'
        return
    image = scipy.misc.imread(FLAGS.image)
    image = scipy.misc.imresize(image, [240, 320])
    if not image.size:
        print('Cannot read image from: %s' % FLAGS.image)
    with tf.Session() as sess:
        # set keep_prob to 1, then it will simply ignore drop_out layer
        saver = tf.train.import_meta_graph('./output/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./output/'))
        pred = sess.run("pred_up:0", feed_dict={"input_image:0":[image],
                                                "keep_prob:0":1.0})
        color_image = cvt_result2color(pred[0], num_classes=2)
        print pred[0].shape
        scipy.misc.imsave('inference.png', color_image)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data_dir', type=str,
                        default='../dataset/cmu_corridor_dataset',
                        help='Directory of input dataset')
    PARSER.add_argument('--data_ext', type=str,
                        default='png',
                        help='File extension of image data')
    PARSER.add_argument('--image', type=str, default=None,
                        help='Image to infer (mendatory)')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
