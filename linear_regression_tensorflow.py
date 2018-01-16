import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)


X_data = np.random.rand(100).astype(np.float32)  #generating sample data
#print(X_data)
y_data = X_data*3 + 2
#print(y_data)
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0,scale=0.1))(y_data)
#print(y_data)

#l = zip(X_data,y_data)[0:5]

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * X_data + b

#minimize the squared error
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


train_data = []

for step in range(100):
    evals = sess.run([train,a,b])[1:]
    if step % 5 ==0:
        print(step,evals)
        train_data.append(evals)

clr = plt.colors
cr,cg,cb = (1.0,1.0,0.0)

for f in train_data:
    cb += 1.0/len(train_data)
    cg -= 1.0/len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a,b] = f
    f_y = np.vectorize(lambda x: a*x + b)(X_data)
    line = plt.plot(X_data,f_y)
    plt.setp(line,color=(cr,cg,cb))

plt.plot(X_data,y_data, 'ro')

g_line = mpatches.Patch(color='red', label = 'Data Points')

plt.legend(handles = [g_line])
plt.show()








