import tensorflow as tf

value = tf.Variable(0,name="value")
one = tf.constant(1)
new_value = tf.add(value,one)
update_value=tf.assign(value,new_value)

initialize_var = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initialize_var)
    print(sess.run(value))
    for _ in range(10):
        sess.run(update_value)
        print(sess.run(value))

"""
>>> 
0
1
2
3
4
5
6
7
8
9
10
>>>
"""     
