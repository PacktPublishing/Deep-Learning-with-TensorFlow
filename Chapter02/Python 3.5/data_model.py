import tensorflow as tf

scalar = tf.constant(100)
vector = tf.constant([1, 2, 3, 4, 5])
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
cube_matrix = tf.constant([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])

print(scalar.get_shape())
print(vector.get_shape())
print(matrix.get_shape())
print(cube_matrix.get_shape())

"""
>>> 
()
(5,)
(2, 3)
(3, 3, 1)
>>> 
"""

