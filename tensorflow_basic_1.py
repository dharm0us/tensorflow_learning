#first cut from https://www.tensorflow.org/get_started/, then modified
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + np.random.random_sample(x_data.shape).astype(np.float32) + 0.3
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
#optimizer = tf.train.RMSPropOptimizer(0.002)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
print(optimizer.__class__)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

x_min = np.amin(x_data)
x_max = np.amax(x_data)
y_min = W.eval(sess)*x_min +b.eval(sess)
y_max = W.eval(sess)*x_max +b.eval(sess)
print(x_min,x_max,y_min,y_max)
plt.plot(x_data,y_data,'ro')
plt.plot([x_min,x_max],[y_min,y_max],'b-')
plt.show(block=True)
