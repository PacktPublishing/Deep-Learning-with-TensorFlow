import numpy as np
import tensorflow as tf
import datetime

log_device_placement = True
n = 10

A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

c1 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

#FIRST GPU
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    c1.append(matpow(a, n))
    
#SECOND GPU
with tf.device('/gpu:1'):
    b = tf.placeholder(tf.float32, [10000, 10000])
    c1.append(matpow(b, n))


with tf.device('/cpu:0'):
    sum = tf.add_n(c1) 
    print(sum)

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)) as sess:
     sess.run(sum, {a:A, b:B})

t2_1 = datetime.datetime.now()
