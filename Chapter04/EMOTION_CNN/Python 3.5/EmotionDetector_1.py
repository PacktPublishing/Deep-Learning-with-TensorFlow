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
        image_size = IMAGE_SIZE // 4
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
    'wf1': weight_variable([(IMAGE_SIZE // 4) * (IMAGE_SIZE // 4) * 64, 256],name="W_fc1"),
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
Training Loss: 1.951450
2017-07-27 14:26:41.689096 Validation Loss: 1.958948
Training Loss: 1.899691
Training Loss: 1.873583
Training Loss: 1.883454
Training Loss: 1.794849
Training Loss: 1.884183
Training Loss: 1.848423
Training Loss: 1.838916
Training Loss: 1.918565
Training Loss: 1.829074
Training Loss: 1.864008
2017-07-27 14:27:00.305351 Validation Loss: 1.790150
Training Loss: 1.753058
Training Loss: 1.615597
Training Loss: 1.571414
Training Loss: 1.623350
Training Loss: 1.494578
Training Loss: 1.502531
Training Loss: 1.349338
Training Loss: 1.537164
Training Loss: 1.364067
Training Loss: 1.387331
2017-07-27 14:27:20.328279 Validation Loss: 1.375231
Training Loss: 1.186529
Training Loss: 1.386529
Training Loss: 1.270537
Training Loss: 1.211034
Training Loss: 1.096524
Training Loss: 1.192567
Training Loss: 1.279141
Training Loss: 1.199098
Training Loss: 1.017902
Training Loss: 1.249009
2017-07-27 14:27:38.844167 Validation Loss: 1.178693
Training Loss: 1.222699
Training Loss: 0.970940
Training Loss: 1.012443
Training Loss: 0.931900
Training Loss: 1.016142
Training Loss: 0.943123
Training Loss: 1.099365
Training Loss: 1.000534
Training Loss: 0.925840
Training Loss: 0.895967
2017-07-27 14:27:57.399234 Validation Loss: 1.103102
Training Loss: 0.863209
Training Loss: 0.833549
Training Loss: 0.812724
Training Loss: 1.009514
Training Loss: 1.024465
Training Loss: 0.961753
Training Loss: 0.986352
Training Loss: 0.959654
Training Loss: 0.774006
Training Loss: 0.858462
2017-07-27 14:28:15.782431 Validation Loss: 1.000128
Training Loss: 0.663166
Training Loss: 0.785379
Training Loss: 0.821995
Training Loss: 0.945040
Training Loss: 0.909402
Training Loss: 0.797702
Training Loss: 0.769628
Training Loss: 0.750213
Training Loss: 0.722645
Training Loss: 0.800091
2017-07-27 14:28:34.632889 Validation Loss: 0.924810
Training Loss: 0.878261
Training Loss: 0.817574
Training Loss: 0.856897
Training Loss: 0.752512
Training Loss: 0.881165
Training Loss: 0.710394
Training Loss: 0.721797
Training Loss: 0.726897
Training Loss: 0.624348
Training Loss: 0.730256
2017-07-27 14:28:53.171239 Validation Loss: 0.901341
Training Loss: 0.685925
Training Loss: 0.630337
Training Loss: 0.656826
Training Loss: 0.666020
Training Loss: 0.627277
Training Loss: 0.698149
Training Loss: 0.722851
Training Loss: 0.722231
Training Loss: 0.701155
Training Loss: 0.684319
2017-07-27 14:29:11.596521 Validation Loss: 0.894154
Training Loss: 0.738686
Training Loss: 0.580629
Training Loss: 0.545667
Training Loss: 0.614124
Training Loss: 0.640999
Training Loss: 0.762669
Training Loss: 0.628534
Training Loss: 0.690788
Training Loss: 0.628837
Training Loss: 0.565587
2017-07-27 14:29:30.075707 Validation Loss: 0.825970
Training Loss: 0.551373
Training Loss: 0.466755
Training Loss: 0.583116
Training Loss: 0.644869
Training Loss: 0.626141
Training Loss: 0.609953
Training Loss: 0.622723
Training Loss: 0.696944
Training Loss: 0.543604
Training Loss: 0.436234
2017-07-27 14:29:48.517299 Validation Loss: 0.873586

>>> 
"""
