import argparse
import sys
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np

FLAGS = None

# constants used for training
learning_rate = 0.005
epochs = 20
batch_size = 512
num_batches = int(mnist.train.num_examples / batch_size)
input_height = 28
input_width = 28
n_classes = 10
dropout = 0.75
display_step = 1
filter_height = 5
filter_width = 5
depth_in = 1
depth_out1 = 64
depth_out2 = 128
dense_ct = 1024


def conv2d(x, W, b, strides=1):
    with tf.name_scope('convolution'):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


def maxpool2d(x, stride=2):
    with tf.name_scope('max_pool'):
        return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, input_height, input_width, 1])
    # convolution
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, stride=2)
    # convolution
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, stride=2)
    # fully connected
    with tf.name_scope('fully_connected1'):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # apply dropout
        fc1 = tf.nn.dropout(fc1, dropout)
    # output prediction
    with tf.name_scope('output'):
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def main(_):
    print(FLAGS.job_name, FLAGS.task_index)
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    print("ps hosts: ", ps_hosts, "worker hosts", worker_hosts)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            # input placeholders
            with tf.variable_scope('input_data'):
                x = tf.placeholder(tf.float32, [None, input_height * input_width])
            with tf.variable_scope('input_labels'):
                y = tf.placeholder(tf.float32, [None, n_classes])
            with tf.variable_scope('keep_prob'):
                keep_prob = tf.placeholder(tf.float32)

            # define weights and biases
            weights = {
                'wc1': tf.Variable(tf.random_normal([filter_height, filter_width, depth_in, depth_out1]),name='w_c1'),
                'wc2': tf.Variable(tf.random_normal([filter_height, filter_width, depth_out1, depth_out2]),name='w_c2'),
                'wd1': tf.Variable(
                    tf.random_normal([int((input_height / 4) * (input_width / 4) * depth_out2), dense_ct]),name='w_d'),
                'out': tf.Variable(tf.random_normal([dense_ct, n_classes]),name='w_out')
            }
            biases = {
                'bc1': tf.Variable(tf.random_normal([depth_out1]),name='b_c1'),
                'bc2': tf.Variable(tf.random_normal([depth_out2]),name='b_c2'),
                'bd1': tf.Variable(tf.random_normal([dense_ct]),name='b_d'),
                'out': tf.Variable(tf.random_normal([n_classes]),name='b_out')
            }
            #log histograms of weights and biases to tensorboard
            for weight in weights:
                with tf.variable_scope(weight):
                    tf.summary.histogram(weight, weights[weight])
            for bias in biases:
                with tf.variable_scope(bias):
                    tf.summary.histogram(bias, biases[bias])
            pred = conv_net(x, weights, biases, keep_prob)
            # create or get global step
            global_step = tf.train.get_or_create_global_step()
            # define loss function and optimizer
            with tf.name_scope('cost'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                tf.summary.scalar('cost', cost)
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
            # evaluate model
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

            #merge all summaries, to be added to scaffold
            merged = tf.summary.merge_all()
            hooks = [
                tf.train.StopAtStepHook(last_step=5000)
            ]
            init = tf.global_variables_initializer()
            #initialize and pass summary op to session using scaffold.
            scaffold = tf.train.Scaffold(init_op=init, init_feed_dict={x:np.zeros(shape=[1,28*28]),y:np.zeros(shape=[1,n_classes])},
                                         summary_op=merged)

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        #config to set low level processor flags, comment to use defaults.
        config = tf.ConfigProto(intra_op_parallelism_threads=21, inter_op_parallelism_threads=5,
                                allow_soft_placement=True) #, device_count={'CPU': 24})
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir=FLAGS.log_dir,
                                               hooks=hooks,
                                               config=config,
                                               save_checkpoint_steps=100,
                                               scaffold=scaffold
                                               ) as mon_sess:
            #low level flags, comment to use defaults.
            # os.environ['OMP_DYNAMIC']='.TRUE.'
            os.environ['OMP_NUM_THREADS'] = '64'
            os.environ['KMP_BLOCKTIME'] = '0'
            os.environ['KMP_SETTINGS'] = '0'
            os.environ['KMP_AFFINITY'] = 'granularity=fine,noverbose,compact,1,0'

            #training loop
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                ### Modified here ###
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, cost_summary, acc_summary = mon_sess.run([train_op,cost,accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                # cost_summary, acc_summary = mon_sess.run([cost, accuracy],
                #                                          feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                print('cost: %s acc: %s' % (cost_summary, acc_summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="Directory for logging"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



