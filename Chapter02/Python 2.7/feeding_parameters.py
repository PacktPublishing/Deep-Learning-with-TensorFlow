import tensorflow as tf
import numpy as np

a = 3
b = 2


x = tf.placeholder(tf.float32,shape=(a,b))
y = tf.add(x,x)

data = np.random.rand(a,b)

sess = tf.Session()

print sess.run(y,feed_dict={x:data})

