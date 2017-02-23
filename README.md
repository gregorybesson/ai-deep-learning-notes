# Regression models

Linear regression
This regression is dedicated to regression of continuous data

Logisitic regression
This regression is dedicated to regression of discrete data

# Neural network

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
