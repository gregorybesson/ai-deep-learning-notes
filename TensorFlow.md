# basic classes

tf.Constant

tf.Placeholder

tf.Variable

tf.Cast

 tf.global_variables_initializer() function to initialize the state of all the Variable tensors:
```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

Linear functions

```y = xW + b```

- W is a matrix of the weights connecting 2 layers.
- y is the output
- x is the input vector
- b is the biases vector

```
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
biases = tf.Variable(tf.zeros(n_labels))
```

The tf.truncated_normal() function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean
The tf.zeros() function returns a tensor with all zeros.

