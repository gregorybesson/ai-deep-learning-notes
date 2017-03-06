# Definition
A convolutional network or convnet is a deep neural network to extract features from images. It is used to detect or identify objects in an image.


## statistical invariants
translation invariance : We want a cat to be recognized wether it's in the left corner of an image or the right one.

Or the word kitten to be recognized wether it's at the begiining or at the end of a sentence.
We'll use the concept of *weight sharing*

ReLU : rectified linear unit.

We'll build a multilayer neural network 



Preventing overfitting

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
