import tensorflow as tf
import numpy as np
#import os, sys, inspect
from datetime import datetime
import EmotionDetectorUtils

"""
lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
"""


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "EmotionDetector/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/EmotionDetector_logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 1001
REGULARIZATION = 1e-2
IMAGE_SIZE = 48
NUM_LABELS = 7
VALIDATION_PERCENT = 0.1


def add_to_regularization_loss(W, b):
    tf.add_to_collection("losses", tf.nn.l2_loss(W))
    tf.add_to_collection("losses", tf.nn.l2_loss(b))

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], \
                          strides=[1, 2, 2, 1], padding="SAME")


def emotion_cnn(dataset):
    with tf.name_scope("conv1") as scope:
        #W_conv1 = weight_variable([5, 5, 1, 32])
        #b_conv1 = bias_variable([32])
        tf.summary.histogram("W_conv1", weights['wc1'])
        tf.summary.histogram("b_conv1", biases['bc1'])
        conv_1 = tf.nn.conv2d(dataset, weights['wc1'],\
                              strides=[1, 1, 1, 1], padding="SAME")
        h_conv1 = tf.nn.bias_add(conv_1, biases['bc1'])
        #h_conv1 = conv2d_basic(dataset, W_conv1, b_conv1)
        h_1 = tf.nn.relu(h_conv1)
        h_pool1 = max_pool_2x2(h_1)
        add_to_regularization_loss(weights['wc1'], biases['bc1'])

    with tf.name_scope("conv2") as scope:
        #W_conv2 = weight_variable([3, 3, 32, 64])
        #b_conv2 = bias_variable([64])
        tf.summary.histogram("W_conv2", weights['wc2'])
        tf.summary.histogram("b_conv2", biases['bc2'])
        conv_2 = tf.nn.conv2d(h_pool1, weights['wc2'], strides=[1, 1, 1, 1], padding="SAME")
        h_conv2 = tf.nn.bias_add(conv_2, biases['bc2'])
        #h_conv2 = conv2d_basic(h_pool1, weights['wc2'], biases['bc2'])
        h_2 = tf.nn.relu(h_conv2)
        h_pool2 = max_pool_2x2(h_2)
        add_to_regularization_loss(weights['wc2'], biases['bc2'])

    with tf.name_scope("fc_1") as scope:
        prob = 0.5
        image_size = IMAGE_SIZE / 4
        h_flat = tf.reshape(h_pool2, [-1, image_size * image_size * 64])
        #W_fc1 = weight_variable([image_size * image_size * 64, 256])
        #b_fc1 = bias_variable([256])
        tf.summary.histogram("W_fc1", weights['wf1'])
        tf.summary.histogram("b_fc1", biases['bf1'])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, weights['wf1']) + biases['bf1'])
        h_fc1_dropout = tf.nn.dropout(h_fc1, prob)
        
    with tf.name_scope("fc_2") as scope:
        #W_fc2 = weight_variable([256, NUM_LABELS])
        #b_fc2 = bias_variable([NUM_LABELS])
        tf.summary.histogram("W_fc2", weights['wf2'])
        tf.summary.histogram("b_fc2", biases['bf2'])
        #pred = tf.matmul(h_fc1, weights['wf2']) + biases['bf2']
        pred = tf.matmul(h_fc1_dropout, weights['wf2']) + biases['bf2']

    return pred

weights = {
    'wc1': weight_variable([5, 5, 1, 32], name="W_conv1"),
    'wc2': weight_variable([3, 3, 32, 64],name="W_conv2"),
    'wf1': weight_variable([(IMAGE_SIZE / 4) * (IMAGE_SIZE / 4) * 64, 256],name="W_fc1"),
    'wf2': weight_variable([256, NUM_LABELS], name="W_fc2")
}

biases = {
    'bc1': bias_variable([32], name="b_conv1"),
    'bc2': bias_variable([64], name="b_conv2"),
    'bf1': bias_variable([256], name="b_fc1"),
    'bf2': bias_variable([NUM_LABELS], name="b_fc2")
}

def loss(pred, label):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))
    tf.summary.scalar('Entropy', cross_entropy_loss)
    reg_losses = tf.add_n(tf.get_collection("losses"))
    tf.summary.scalar('Reg_loss', reg_losses)
    return cross_entropy_loss + REGULARIZATION * reg_losses


def train(loss, step):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=step)


def get_next_batch(images, labels, step):
    offset = (step * BATCH_SIZE) % (images.shape[0] - BATCH_SIZE)
    batch_images = images[offset: offset + BATCH_SIZE]
    batch_labels = labels[offset:offset + BATCH_SIZE]
    return batch_images, batch_labels


def main(argv=None):
    train_images, train_labels, valid_images, valid_labels, test_images = EmotionDetectorUtils.read_data(FLAGS.data_dir)
    print("Train size: %s" % train_images.shape[0])
    print('Validation size: %s' % valid_images.shape[0])
    print("Test size: %s" % test_images.shape[0])

    global_step = tf.Variable(0, trainable=False)
    dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1],name="input")
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])

    pred = emotion_cnn(input_dataset)
    output_pred = tf.nn.softmax(pred,name="output")
    loss_val = loss(pred, input_labels)
    train_op = train(loss_val, global_step)

    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph_def)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model Restored!")

        for step in range(MAX_ITERATIONS):
            batch_image, batch_label = get_next_batch(train_images, train_labels, step)
            feed_dict = {input_dataset: batch_image, input_labels: batch_label}

            sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                train_loss, summary_str = sess.run([loss_val, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                print("Training Loss: %f" % train_loss)

            if step % 100 == 0:
                valid_loss = sess.run(loss_val, feed_dict={input_dataset: valid_images, input_labels: valid_labels})
                print("%s Validation Loss: %f" % (datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=step)


if __name__ == "__main__":
    tf.app.run()



"""
>>> 
Train size: 3761
Validation size: 417
Test size: 1312
WARNING:tensorflow:Passing a `GraphDef` to the SummaryWriter is deprecated. Pass a `Graph` object instead, such as `sess.graph`.
Training Loss: 1.962236
2016-11-05 22:39:36.645682 Validation Loss: 1.962719
Training Loss: 1.907290
Training Loss: 1.849100
Training Loss: 1.871116
Training Loss: 1.798998
Training Loss: 1.885601
Training Loss: 1.849380
Training Loss: 1.843139
Training Loss: 1.933691
Training Loss: 1.829839
Training Loss: 1.839772
2016-11-05 22:42:58.951699 Validation Loss: 1.822431
Training Loss: 1.772197
Training Loss: 1.666473
Training Loss: 1.620869
Training Loss: 1.592660
Training Loss: 1.422701
Training Loss: 1.436721
Training Loss: 1.348217
Training Loss: 1.432023
Training Loss: 1.347753
Training Loss: 1.299889
2016-11-05 22:46:55.144483 Validation Loss: 1.335237
Training Loss: 1.108747
Training Loss: 1.197601
Training Loss: 1.245860
Training Loss: 1.164120
Training Loss: 0.994351
Training Loss: 1.072356
Training Loss: 1.193485
Training Loss: 1.118093
Training Loss: 1.021220
Training Loss: 1.069752
2016-11-05 22:50:17.677074 Validation Loss: 1.111559
Training Loss: 1.099430
Training Loss: 0.966327
Training Loss: 0.960916
Training Loss: 0.844742
Training Loss: 0.979741
Training Loss: 0.891897
Training Loss: 1.013132
Training Loss: 0.936738
Training Loss: 0.911577
Training Loss: 0.862605
2016-11-05 22:53:30.999141 Validation Loss: 0.999061
Training Loss: 0.800337
Training Loss: 0.776097
Training Loss: 0.799260
Training Loss: 0.919926
Training Loss: 0.758807
Training Loss: 0.807968
Training Loss: 0.856378
Training Loss: 0.867762
Training Loss: 0.656170
Training Loss: 0.688761
2016-11-05 22:56:53.256991 Validation Loss: 0.931223
Training Loss: 0.696454
Training Loss: 0.725157
Training Loss: 0.674037
Training Loss: 0.719200
Training Loss: 0.749460
Training Loss: 0.741768
Training Loss: 0.702719
Training Loss: 0.734194
Training Loss: 0.669155
Training Loss: 0.641528
2016-11-05 23:00:06.530139 Validation Loss: 0.911489
Training Loss: 0.764550
Training Loss: 0.646964
Training Loss: 0.724712
Training Loss: 0.726692
Training Loss: 0.656019
Training Loss: 0.690552
Training Loss: 0.537638
Training Loss: 0.680097
Training Loss: 0.554115
Training Loss: 0.590837
2016-11-05 23:03:15.351156 Validation Loss: 0.818303
Training Loss: 0.656608
Training Loss: 0.567394
Training Loss: 0.545324
Training Loss: 0.611726
Training Loss: 0.600910
Training Loss: 0.526467
Training Loss: 0.584986
Training Loss: 0.567015
Training Loss: 0.555465
Training Loss: 0.630097
2016-11-05 23:06:26.575298 Validation Loss: 0.824178
Training Loss: 0.662920
Training Loss: 0.512493
Training Loss: 0.475912
Training Loss: 0.455112
Training Loss: 0.567875
Training Loss: 0.582927
Training Loss: 0.509225
Training Loss: 0.602916
Training Loss: 0.521976
Training Loss: 0.445122
2016-11-05 23:09:40.136353 Validation Loss: 0.803449
Training Loss: 0.435535
Training Loss: 0.459343
Training Loss: 0.481706
Training Loss: 0.460640
Training Loss: 0.554570
Training Loss: 0.427962
Training Loss: 0.512764
Training Loss: 0.531128
Training Loss: 0.364465
Training Loss: 0.432366
2016-11-05 23:12:50.769527 Validation Loss: 0.851074
>>> 
"""
