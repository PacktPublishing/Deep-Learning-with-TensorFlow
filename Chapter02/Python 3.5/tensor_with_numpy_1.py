import tensorflow as tf
import numpy as np

#tensore 1d con valori costanti
tensor_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
tensor_1d = tf.constant(tensor_1d)
with tf.Session() as sess:
    print(tensor_1d.get_shape())
    print(sess.run(tensor_1d))

#tensore 2d con valori variabili
tensor_2d = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
tensor_2d = tf.Variable(tensor_2d)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tensor_2d.get_shape())
    print(sess.run(tensor_2d))


tensor_3d = np.array([[[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8]],
                      [[ 9, 10, 11],[12, 13, 14],[15, 16, 17]],
                      [[18, 19, 20],[21, 22, 23],[24, 25, 26]]])

tensor_3d = tf.convert_to_tensor(tensor_3d, dtype=tf.float64)
with tf.Session() as sess:
    print(tensor_3d.get_shape())
    print(sess.run(tensor_3d))


interactive_session = tf.InteractiveSession()
tensor = np.array([1, 2, 3, 4, 5])
tensor = tf.constant(tensor)
print(tensor.eval())
interactive_session.close()

"""
Python 2.7.10 (default, Oct 14 2015, 16:09:02) 
[GCC 5.2.1 20151010] on linux2
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 
(10,)
[ 1  2  3  4  5  6  7  8  9 10]
(3, 3)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
(3, 3, 3)
[[[  0.   1.   2.]
  [  3.   4.   5.]
  [  6.   7.   8.]]

 [[  9.  10.  11.]
  [ 12.  13.  14.]
  [ 15.  16.  17.]]

 [[ 18.  19.  20.]
  [ 21.  22.  23.]
  [ 24.  25.  26.]]]
[1 2 3 4 5]
>>> 
"""


