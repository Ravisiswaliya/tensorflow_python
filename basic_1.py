import tensorflow as tf #importing tensor flow module

a = tf.constant([2])  #defining constant
b = tf.constant([5])  #defining constant

c = tf.add(a,b)  #using tensorflow function

with tf.Session() as session:   #using session
    result = session.run(c)
    print(result)               #addition of two constant