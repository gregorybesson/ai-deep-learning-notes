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

To determine the number of neurons we need to connect to a patch, it depends on the flter depth (k). Having multiple neurons for a given patch ensure that our CNN can learn to capture whatever characteristics the CNN learns are important.
The volume of the output layer is:
(W-F+2P)/S + 1

S: Stride
W: Volume of our input layer
F: Volume of our Filter (= height * width * depth)
P: Padding

Hyper-parameters:
- Filter width and height
- stride
- filter depth k

## Preventing overfitting

- Early termination: We observe the performance of pur model on our validation set and stop when there is no improvement anymore.

- Regularization: Applying artificial constraints on my network reducing implicitly the number of free parameters. 
  - L2 regularization: We add another term to the loss, which penalize large weights (we add the L2 norm of the weight to the loss multiplied by a small constant)
  - Dropout: The values that come from one layer to another one are called activations. Let's set randomly half of them  to zero and scale the remaining activations by a factor of 2. It makes your model creating redundant representations, which makes your model more robust and prevents overfitting. Dropout produces an ensemble of training. As we want the consensus when we evaluate the model. We do average the activations


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
