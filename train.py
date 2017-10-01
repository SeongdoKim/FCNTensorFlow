from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

import scipy
import scipy.misc
import fcn_vgg32
import cmu_corridor_dataset_parser as data_parser
import FCNDatsetReader as Reader

FLAGS = None

def cvt_result2color(image, num_classes=20):
    """
    Convert given inferenced label image to color image
    """
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def main(_):
    """
    Main entry for FCN training
    """
    # variables
    num_classes = FLAGS.classes
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    image = tf.placeholder(tf.float32, shape=[None, 240, 320, 3], name="input_image")
    labels = tf.placeholder(tf.int32, shape=[None, 240, 320, 1], name="annotation")

    # construct a fully convolutional networks
    fcn = fcn_vgg32.FCNVGG32(image,
                             keep_prob=keep_prob,
                             num_classes=num_classes,
                             weights_path="./vgg16.npy")

    anno_pred, logits = fcn.build(train=FLAGS.train)
    #tf.summary.image("input_image", image, max_outputs=2)
    #tf.summary.image("ground_truth", tf.cast(labels, tf.uint8), max_outputs=2)
    #tf.summary.image("pred_annotation", tf.cast(anno_pred, tf.uint8), max_outputs=2)

    # construct an optimizer
    trainable_var = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    loss = tf.reduce_mean((
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.squeeze(labels, squeeze_dims=[3]),
                                                       name="entropy")))

    gradients = optimizer.compute_gradients(loss, var_list=trainable_var)

    tf.summary.scalar("entropy", loss)

    train_op = optimizer.apply_gradients(gradients)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # Read dataset
    reader = Reader.FCNDatsetReader(data_parser.read_dataset(FLAGS.data_dir, FLAGS.data_ext))
    print("%d images and labels are in the dataset" % len(reader.filelist))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Construct a session and run the optimization
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # initialize global variables
        # set log output directory
        graph_location = './log/' #tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())
        if FLAGS.train:
            for step in range(FLAGS.max_iter):
                train_images, train_annotations = reader.next_batch(FLAGS.batch_size)
                feed_dict = {image: train_images,
                             labels: train_annotations,
                             keep_prob: FLAGS.keep_prob}
                sess.run(train_op, feed_dict=feed_dict)
                if step % 100 == 0: # print accuracy for every 100 iteration
                    train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    print('Step %d, train loss:%g' % (step, train_loss))
                    train_writer.add_summary(summary_str, step)
                if step % 5000 == 0: # save model for every 10000 iteration
                    saver.save(sess, FLAGS.output, global_step=step)
        else:
            valid_images, valid_annotations = reader.get_random_batch(2)
            #valid_images = [scipy.misc.imresize(scipy.misc.imread("./tabby_cat.png"),
            #                                        [240, 320], interp='nearest')]
            feed_dict = {image: valid_images, labels: valid_annotations}
            anno_pred, logits = sess.run([anno_pred, logits], feed_dict=feed_dict)
            color_image = cvt_result2color(anno_pred[0], num_classes=num_classes)
            scipy.misc.imsave('inference.png', color_image)
            scipy.misc.imsave('original.png', valid_images[0])
        # Save the trained variables to disk.
        save_path = saver.save(sess, FLAGS.output)
        print("Model saved in file: %s" % save_path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data_dir', type=str,
                        default='../dataset/cmu_corridor_dataset',
                        help='Directory of input dataset')
    PARSER.add_argument('--data_ext', type=str,
                        default='png',
                        help='File extension of image data')
    PARSER.add_argument('--classes', type=int,
                        default=2,
                        help='The number of classes')
    PARSER.add_argument('--output', type=str,
                        default='./output/model',
                        help='Full path including prefix of output file')
    PARSER.add_argument('--max_iter', type=int,
                        default=10000,
                        help='Maximum iteration of training')
    PARSER.add_argument('--batch_size', type=int,
                        default=8,
                        help='Training batch size')
    PARSER.add_argument('--learning_rate', type=float,
                        default=1e-6, help='Initial learning rate')
    PARSER.add_argument('--keep_prob', type=float,
                        default=0.5, help='Dropout probability')
    PARSER.add_argument('--train', type=bool,
                        default=False, help='Training/Evaluation')
    PARSER.add_argument('--debug', type=bool,
                        default=True, help='Print out training process is set to True')

    FLAGS, UNPARSED = PARSER.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
