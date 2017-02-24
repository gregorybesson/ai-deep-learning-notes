# hello world
```
import tensorflow as tf

x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

# basic classes
- tf.constant
- tf.placeholder
- tf.Variable
- tf.Session

Operations:
- tf.add()
- tf.subtract()
- tf.multiply()
- tf.divide()
- tf.matmul(): matrix multiplication
- tf.cast
- feed_dict: in tf.session.run() ```output = sess.run(x, feed_dict={x: 'Hello World'})```
- tf.global_variables_initializer() function to initialize the state of all the Variable tensors:
```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```
- tf.truncated_normal(): returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean
- tf.zeros(): returns a tensor with all zeros.
```
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
biases = tf.Variable(tf.zeros(n_labels))
```

Linear functions

```y = xW + b```

- W is a matrix of the weights connecting 2 layers.
- y is the output
- x is the input vector
- b is the biases vector





