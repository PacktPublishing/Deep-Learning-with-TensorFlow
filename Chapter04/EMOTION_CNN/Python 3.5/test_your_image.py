from scipy import misc
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import os, sys, inspect
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import EmotionDetectorUtils

emotion = {0:'anger', 1:'disgust',\
           2:'fear',3:'happy',\
           4:'sad',5:'surprise',6:'neutral'}


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('author_img.jpg')     
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()


""""
lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
"""



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "EmotionDetector/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/EmotionDetector_logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")




train_images, train_labels, valid_images, valid_labels, test_images = \
                  EmotionDetectorUtils.read_data(FLAGS.data_dir)


sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('logs/model.ckpt-1000.meta')
new_saver.restore(sess, 'logs/model.ckpt-1000')
tf.get_default_graph().as_graph_def()

x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")

image_0 = np.resize(gray,(1,48,48,1))

result = sess.run(y_conv, feed_dict={x:image_0})
label = sess.run(tf.argmax(result, 1))
print(emotion[label[0]])



