# Definition
A convolutional network or convnet is a deep neural network to extract features from images. It is used to detect or identify objects in an image.
Basically, a CNN will learn to recognize basic lines and curves, then shpes and blobs, then increasingly complex objects in the image. Finally, the CNN classifies the image by combining the larger, more complex objects.

With deep learning, the CNN learns by itself these specific features htrough forward propagation and backpropagation.

We'll build a CNN with multiple convolutional layers using the stride to reduce the dimensionality and increasing the filter depth layer after layer. Once we obtain a deep and narrow neural network, we connect fully connected layers then a classifier.


## statistical invariants
translation invariance : We want a cat to be recognized wether it's in the left corner of an image or the right one.

Or the word kitten to be recognized wether it's at the beginning or at the end of a sentence.
We'll use the concept of *weight sharing*
The weights and patches we learn for a given output layer are shared across all patches in a given input layer.

We'll build a multilayer neural network 

# Creating a covnet
## Breaking up the image
### Patches
We extract *patches* or *kernels* from an image. The depth being the colors. Each layer of the image is called a *feature map*.
To parse the whole image, we'll use a *stride* to shift patches from stride pixels at a time.
valid padding: we stay inside the image
same padding: we add zeros to the border of the image => The output map size will be exactly the same as the input map size.

### Filter
we define a width and height that defines a *filter* that will looks at patches of the image (patches are the same size as the filter). The filter groups together adjacent pixels and treats them as collectve. This is because adjacent pixels have a special meaning to each other. Our CNN will learn how to classify local patterns.

We can define a filter depth which is a set of different filters to improve the efficiency of the CNN.

To determine the number of neurons we need to connect to a patch, it depends on the filter depth (k). Having multiple neurons for a given patch ensure that our CNN can learn to capture whatever characteristics the CNN learns are important.
The volume of the output layer is:
(W-F+2P)/S + 1

S: Stride
W: Volume of our input layer
F: Volume of our Filter (= height * width * depth)
P: Padding

The total of parameters (neurons) of the conv layer will be:
(Hf * Wf * Df + 1) * (Ho * Wo * Do)

Hf, Wf, Df: dimensions of the filter +1 for the bias
Ho, Wo, Do: dimensions of the conv ooutput

We see with this formula that each weight is assigned with each single part of the output.

### Number of parameters of a convolutional layer
instead of making our neurons share their parameters with ALL other neurons of all Channels, we'll take the number of parameters in the convolutional layer, if every neuron in the output layer shares its parameters with every other neuron in its *same channel* (the same depth).

(Hf * Wf * Df + 1) * Do


Hyper-parameters:
- Filter width and height
- stride
- filter depth k

## The convolutional layer
```
# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```
## Improving the convolutional layer

The stride method is very information destructive. We could instead use pooling which take all the convolutions in the neighborhood and combine them.

### Max-Pooling
y = max(Xi) We take the max of a neighborhood.
Conceptually, the benefit of the max pooling operation is to reduce the size of the input, and allow the neural network to focus on only the most important elements. 

ie. with a 2x2 filter and stride of 2:
[[1, 0], [4, 6]] becomes 6, because 6 is the maximum value in this set

A typical conv network is:
- Image
- Convolution
- Max Pooling
- Convolution
- Max Pooling
- Fully connected
- Fully connected
- Classifier

### Average Pooling
y = mean(Xi) We take the average of a neighborhood.

These methods have lost ground and are replaced by the dropout method, more reliable and efficient.

### Preventing overfitting

- Early termination: We observe the performance of pur model on our validation set and stop when there is no improvement anymore.

- Regularization: Applying artificial constraints on my network reducing implicitly the number of free parameters. 
  - L2 regularization: We add another term to the loss, which penalize large weights (we add the L2 norm of the weight to the loss multiplied by a small constant)
  - Dropout: The values that come from one layer to another one are called activations. Let's set randomly half of them  to zero and scale the remaining activations by a factor of 2. It makes your model creating redundant representations, which makes your model more robust and prevents overfitting. Dropout produces an ensemble of training. As we want the consensus when we evaluate the model. We do average the activations

### Inception modules
Instead od choosing a max-pool layer or 1X1 conv, or 3X3, or 5X5 ... let's use them all. A the top, we just do the composition of all the results

# Step1: Preprocessing data
## Normalize images
## Normalize labels
We do hot-encode labels
## Randomize data
## Preprocess data
We do split data into training, validation and testing sets.

# Step2: Build the network
## Create placeholders
for input, labels and dropout
## Convolution and max pooling layer
## Flatten layer
## Fully-connected layer
## Output layer

# Step3: Create the convolutional model
Apply n convolutional layers + a flatten layer + n' fully connected layers + output layer

# Step4: Train the neural network
# Step5: Test the model
