# Definition
A convolutional network or convnet is a deep neural network to extract features from images. It is used to detect or identify objects in an image.

ReLU : rectified linear unit.

We'll build a multilayer neural network 


Preventing overfitting

- Early termination: We observe the performance of pur model on our validation set and stop when there is no improvement anymore.

- Regularization: Applying artificial constraints on my network reducing implicitly the number of free parameters. 
  - L2 regularization: We add another term to the loss, which penalize large weights (we add the L2 norm of the weight to the loss multiplied by a small constant)
  - Dropout: The values that come from one layer to another one are called activations. Let's set randomly half of them  to zero and scale the remaining activations by a factor of 2. It makes your model creating redundant representations, which makes your model more robust and prevents overfitting. Dropout produces an ensemble of training. As we want the consensus when we evaluate the model. We do average the activations
