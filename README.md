# Regression and classification models

Linear regression
This regression is dedicated to regression of continuous data

Logisitic regression
This regression is dedicated to regression of discrete data

## Definition
regression returns a value
classification returns a state

# Neural network

## introduction
A good neural network generalizes accurately. It can predict a value with accuracy.
We must avoid overfitting (the model is tied to the data it learned on. It has memorized it too much).
We must avoid underfitting too as it oversimplify a model.

## Perceptron or neuron
This is the basic unit of a neural network.
It is a type of linear classifier: it decides wetehr an input represented by a vector of numbers, belongs to some class or not.

An input data coming into a perceptron is multiplied by a weight.
With many data, you'll train your neural network.
The perceptron applies these weights to the inputs and sums them.Ths process is called linear combination.

## Activation function
To turn the perceptron sum into an ouotput signal, we use activation functions.

### Heaviside step function
f(x) = 0 if x < 0
f(x) = 1 if x >= 0

### Sigmoïd function
sigmoid(x) = 1 / (1+e-x)

derivative : 
sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))

## Gradient descent

Weight update
```Δwij = ηδjxi``` 

## Error calculation
### ZSSE
The error term δ:
```δ=(y− ^y)f′(h)=(y−y^)f′(∑wixi)```
- y - ^y is the output error
- f'(h) is the derivative of the activation function

### Mean square error
E =  (1/2m) * ∑(y^mu - y^mu)^2

## Training / validation / test set

we train a model then test it

### sklearn
train_test_split(x, y, test_size)

## Confusion matrix
It helps to know how good a model is.
it stores values:
- true positive
- true negative
- false positive
- false negative

Accuracy = the number of true predictions / all predictions

### sklearn
accuracy_score(y_true, y_pred)
mean_squared_error
mean_absolute_error(y, guesses
r2_score(y_true, y_pred)

