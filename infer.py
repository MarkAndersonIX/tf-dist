import argparse
import sys
import tensorflow as tf
import numpy as np
import cv2
import os
import cnn

FLAGS = None

def main(_):
    if not os.path.exists(FLAGS.image_path):
        raise Exception('Image does not exist!')
    image = cv2.imread(FLAGS.image_path)
    image = image.reshape([1,480,640,3])
    assert image.shape == (1,480,640,3)
    label = np.zeros(11).reshape([1,11])
    #get output layer from model.
    output = cnn.pred

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    #saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            #create a saver object and load the latest checkpoint
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('logs/'))
            res = sess.run(output, feed_dict={cnn.x: image, cnn.y: label, cnn.keep_prob: 1.0})
            print(sess.run(tf.argmax(res,1)))
        except Exception as ex:
            print(ex)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--image_path",
        type=str,
        default="c2.jpg",
        help="Test image path, 480x640x3"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


