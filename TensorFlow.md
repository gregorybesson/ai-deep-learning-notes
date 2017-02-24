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
- tf.log()
- tf.cast()
- tf.reduce_sum(): ```x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15```
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

# Calculating the output of the perceptrons
Linear function

```y = xW + b```

- W is a matrix of the weights connecting 2 layers.
- y is the output
- x is the input vector
- b is the biases vector

# Transforming the output into a probability distribution
Activation function softmax

The softmax function squashes it's inputs, typically called logits or logit scores, to be between 0 and 1 and also normalizes the outputs such that they all sum to 1. This means the output of the softmax function is equivalent to a categorical probability distribution.
```
x = tf.nn.softmax([2.0, 1.0, 0.2])
```

example:
```
import tensorflow as tf

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    softmax = tf.nn.softmax(logits)   
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})
    
    return output
```

# transforming the labels into vectors
Transforming the labels into one-hot encoded vectors is done with scikit-learn using LabelBinarizer.

example:
```
import numpy as np
from sklearn import preprocessing

# Example labels
labels = np.array([1,5,3,2,1,4,2,1,3])

# Create the encoder
lb = preprocessing.LabelBinarizer()

# Here the encoder finds the classes and assigns one-hot vectors 
lb.fit(labels)

# And finally, transform the labels into one-hot encoded vectors
lb.transform(labels)
>>> array([[1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0]])
```

# Classify the results
we use cross entropy as the cost function for classification of one-hot encoded labels.



example:
```
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

x_entropy = -tf.reduce_sum(tf.mul(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    output = sess.run(x_entropy, feed_dict={softmax:softmax_data, one_hot:one_hot_data})
    print (output)
```

# stochastic gradient descent
How to find the correct weights and biases which will predict our results with accuracy ?
That is have a low distance for the correct class, but high distance for the wrong ones.

We may mesure our training loss (the distance average by all weights and biases)
[image]

And we'll try to minimize that function: gradient descent !

## why stochastic
Computing the gradient descent from the training loss is an iterative process on potentially huge data set. it may be very long.
We'll use a random little portion of the training data to measure the average loss and then will compute derivative of this average loss many many times (stochastic gradient descent) which will be more efficient than the regular gradient descent.

But because SGD (Stochastic Gradient Descent) is prone to errors, we have to carefully prepare our inputs and initial weights.
- Inputs should have a mean = 0 and equal variance (small).
- Weights should be randomized, a mean = 0 and equal variance (small).

we have also 2 techniques to minimize errors during SGD:

## Momentum technique
- running average of the direction of previous steps of the sgd

## Learning rate decay
- for each step of the sgd, we decrease the learning rate

## hyper-parameters of a sgd
- initial learning rate
- learning rate decay
- momentum
- batch size
- weight initialization

These numerous hyper-parameters make the optimization of the deep-learning function hard. We then can use
one approach calles ADAGRAD which takes care of initial learning rate, learning rate decay and momentum. It makes learning less sensitive to hyper-parameters.

> SGD is the core of deep learning because it's efficient with big data and big models

