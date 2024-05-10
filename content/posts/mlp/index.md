---
title: "Multilayer perceptron"
date: 2024-05-09T12:00:00Z
draft: true
layout: "single"
mathjax: true
---

# Overview

Goal: Explain how multilayer perceptron model is trained. 

Half of the battle is the configurations and notations:
- Suppose we want to classify 25x25 images and have 10 output labels $(\textbf{y})$. The data for training are labeled and cleaned. 
- There are $m$ instances in the data, 625 features, and 10 labels. 
- The data can be represented as a $m \times 625$ matrix or tabular-type. For simplicity, we will work with each single training example (i.e. some $x_0$). That is, the input layer is a $1 \times 625$. As a convention (I believe), we should transpose this matrix and get the $625 \times 1$ input layer. Then, the output layer is the activations of 10 labels which is a $10 \times 1$ matrix. Output layer notation: $\textbf{a}^{(3)} = [ [a_0^{(3)}], [a_1^{(3)}], \ldots, [a_{9}^{(3)}]]$. To be more general, let $j$ index the output matrix's elements. 
- Choose 2 hidden layers: the first layer has 5 neurons and uses ReLU as the activation function and the second layer has 8 neurons and uses a softmax function. The number of neurons is arbitrarily selected. Respective layer notations: $\textbf{a}^{(1)} = [[a_0^{(1)}], [a_1^{(1)}], \ldots, [a_4^{(1)}]]$ and $\textbf{a}^{(2)} = [[a_0^{(2)}], [a_1^{(2)}], \ldots, [a_7^{(2)} ]]$. To be more general, let $k$ index the layer (2) matrix's elements and let $h$ index the layer (1) matrix's elements. 
- Choose the Mean Squared Error (MSE) loss function. For one single training example $x_i$, $ℒ_i = \sum_{j=0}^{9} (a_j^{(3)} - y_j)^{2}$. Then, the overall loss function for all training examples is $ℒ = \frac{1}{m} \sum_{i=0}^{m-1} \mathscr{L_i}$.
- Choose the Stochastic Gradient Descent technique for mini-batching and randomization optimization. Iterate through each training example (or randomly selected mini-batches of examples) and compute the gradient of the loss function with respect to the $W$ and $b$ parameters using that example or mini-batch. Update the parameters with the computed gradients and a learning rate. Repeat the process for multiple iterations (epochs) until convergence or a stopping criterion is met.
- Last but not least: weight and bias parameters. They appear in each of the layers except for input layer. 
    - $\textbf{W}^{(1)}$ and $\textbf{b}^{(1)}$ denote the weights and biases of layer (1) in which $\textbf{W}^{(1)}$ is a $5 \times 625$ matrix and $\textbf{b}^{(1)}$ is a $5 \times 1$ matrix. 
    - $\textbf{W}^{(2)}$ and $\textbf{b}^{(2)}$ denote the weights and biases of layer (2) in which $\textbf{W}^{(2)}$ is a $8 \times 5$ matrix and $\textbf{b}^{(2)}$ is a $8 \times 1$ matrix. 
    - $\textbf{W}^{(3)}$ and $\textbf{b}^{(3)}$ denote the weights and biases of output layer in which $\textbf{W}^{(3)}$ is a $10 \times 8$ matrix and $\textbf{b}^{(3)}$ is a $10 \times 1$ matrix.   

This is my attempt to draw a neural network for one single training example:

![](/mlp/nn.png)

# Additional notations

- Weighted sum $z$
- ReLU and softmax
- SGD, regularization

# Algorithms 

Repeating over epochs:
- Feed forward
- Back prop
- Update

# Back propagation

# Code: Heart disease classification

Information on the data [here](https://archive.ics.uci.edu/dataset/45/heart+disease.). 


```python
# libraries
import pandas as pd
import numpy as np
```


```python
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# variable information 
heart_disease.variables
```


```python
heart_disease["data"]

```


```python
y
```

# References:

Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
