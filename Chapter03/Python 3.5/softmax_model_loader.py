import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot=True)
sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph('saved_mnist_cnn.ckpt.meta')
new_saver.restore(sess, 'saved_mnist_cnn.ckpt')
tf.get_default_graph().as_graph_def()
x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")
image_b = mnist.test.images[100]
result = sess.run(y_conv, feed_dict={x:image_b})
print(result)
print(sess.run(tf.argmax(result, 1)))

plt.imshow(image_b.reshape([28, 28]), cmap='Greys')
plt.show()





