import argparse
import sys
import os
import tensorflow as tf
import numpy as np

FLAGS = None

#input function to read from TFRecord database (created using tf inception's build_image_data.py)
def dataset_input_fn(dir=os.getcwd(), prefix='train-'):
    filenames = [dir+'/'+f for f in os.listdir(dir) if f.startswith(prefix)]
    print(f for f in filenames)
    if len(filenames) < 1:
        raise Exception("No files found with prefix "+prefix)
    dataset = tf.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/class/label": tf.FixedLenFeature((), tf.int64,
                                                    default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image/encoded"])
        image = tf.reshape(image, [480, 640, 3])
        label = tf.cast(parsed["image/class/label"], tf.int32)
        label = tf.one_hot(label,11)
        return image, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    #features, labels = iterator.get_next()
    return iterator.get_next() #features, labels

def conv2d(x,W,b,strides=2):
    with tf.name_scope('convolution'):
        x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
        x = tf.nn.bias_add(x,b)
        return tf.nn.relu(x)
def maxpool2d(x,stride=2):
    with tf.name_scope('max_pool'):
        return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')
def conv_net(x,weights,biases,dropout):
    #convolution
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, stride=2)
    #convolution
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, stride=2)
    #convolution
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=1)
    conv3 = maxpool2d(conv3, stride=2)
    #fully connected
    with tf.name_scope('fully_connected1'):
        fc1 = tf.reshape(conv3,[-1,weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        #apply dropout
        fc1 = tf.nn.dropout(fc1,dropout)
    #fully connected
    with tf.name_scope('fully_connected2'):
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)
        #apply dropout
        fc2 = tf.nn.dropout(fc2,dropout)
    #output prediction
    with tf.name_scope('output'):
        out = tf.add(tf.matmul(fc2,weights['w_out']),biases['b_out'])
    return out

learning_rate = 0.005
epochs = 200
batch_size = 64
num_batches = int(24000/batch_size) #replace with n/batch_size
input_height = 480
input_width = 640
n_classes = 11
dropout = 0.5
display_step = 1
filter_height = 3
filter_width = 3
depth_in = 3
depth_out1 = 16
depth_out2 = 32
depth_out3 = 64
dense_ct = 128





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
            x = tf.placeholder(tf.float32, [None, input_height, input_width, depth_in])
            y = tf.placeholder(tf.float32, [None, n_classes])
            keep_prob = tf.placeholder(tf.float32)

            # define weights and biases
            weights = {
                'wc1': tf.Variable(tf.random_normal([filter_height + 2, filter_width + 2, depth_in, depth_out1]), name='wc1'),
                'wc2': tf.Variable(tf.random_normal([filter_height + 2, filter_width + 2, depth_out1, depth_out2]), name='wc2'),
                'wc3': tf.Variable(tf.random_normal([filter_height + 4, filter_width + 4, depth_out2, depth_out3]), name='wc3'),
                'wd1': tf.Variable(
                    tf.random_normal([int((input_height / 32) * (input_width / 32) * depth_out3), dense_ct]), name='wd1'),
                'wd2': tf.Variable(tf.random_normal([dense_ct, dense_ct]), name='wd2'),
                'w_out': tf.Variable(tf.random_normal([dense_ct, n_classes]), name='w_out')
            }
            biases = {
                'bc1': tf.Variable(tf.random_normal([depth_out1]), name='bc1'),
                'bc2': tf.Variable(tf.random_normal([depth_out2]), name='bc2'),
                'bc3': tf.Variable(tf.random_normal([depth_out3]), name='bc3'),
                'bd1': tf.Variable(tf.random_normal([dense_ct]), name='bd1'),
                'bd2': tf.Variable(tf.random_normal([dense_ct]), name='bd2'),
                'b_out': tf.Variable(tf.random_normal([n_classes]), name='b_out')
            }
            #log histograms of weights and biases to tensorboard
            for weight in weights:
                with tf.variable_scope(weight):
                    tf.summary.histogram(weight, weights[weight])
            for bias in biases:
                with tf.variable_scope(bias):
                    tf.summary.histogram(bias, biases[bias])

            #create convolutional network
            pred = conv_net(x, weights, biases, keep_prob)

            #create or get global step
            global_step = tf.train.get_or_create_global_step()

            # define loss function and optimizer
            with tf.variable_scope('cost'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
            tf.summary.scalar('loss', cost)
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
            # evaluate model
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            with tf.variable_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            # initialization op
            init = tf.global_variables_initializer()

            merged = tf.summary.merge_all()
            hooks = [
                tf.train.StopAtStepHook(last_step=100000),
                #tf.train.SummarySaverHook(save_steps=10,output_dir=FLAGS.log_dir, summary_op=merged)
            ]
        # cnn dataset (May need to factor this upward)
        iter = dataset_input_fn()

        scaffold = tf.train.Scaffold(init_op=init,
                                     saver=tf.train.Saver(max_to_keep=4),
                                     init_feed_dict={x: np.zeros([1,input_height,input_width,depth_in], dtype=np.float32),
                                                     y: np.zeros([1,n_classes], dtype=np.float32), keep_prob: 1.0},
                                     summary_op=merged)

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir=FLAGS.log_dir,
                                               hooks=hooks,save_summaries_secs=300,
                                               save_checkpoint_secs=300,
                                               scaffold=scaffold
                                               ) as mon_sess:
            #if worker is chief, set up filewriter to save summaries
            # if(FLAGS.task_index == 0):
            #     filewriter = tf.summary.FileWriter(logdir=FLAGS.log_dir)

            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                ### Modified here ###
                images, labels = mon_sess.run(iter)
                mon_sess.run(train_op, feed_dict={x: images, y: labels, keep_prob: dropout})
                cost_summary, acc_summary = mon_sess.run([cost, accuracy], feed_dict={x: images, y: labels, keep_prob: dropout})
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


