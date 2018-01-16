import tensorflow as tf

a = tf.Variable(1)  #define variable

a1 = tf.constant(2) #define constant

n = tf.add(a,a1)  #new value
u = tf.assign(a,n)  #to updating value

#we have to initialized
intlz = tf.global_variables_initializer()
#print(intlz)

with tf.Session() as session:
    session.run(intlz)
    print(session.run(a))
    for _ in range(5):
        session.run(u)
        print(session.run(a))



#using placeholder
ph = tf.placeholder(tf.float32)
b = ph*2

with tf.Session() as ssn:
    result = ssn.run(b,feed_dict={ph:3.5})
    print(result)