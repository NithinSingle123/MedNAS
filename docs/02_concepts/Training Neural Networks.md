---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---


So we saw at any particular layer how we have the data coming in, we multiply by weights and pass it through an activation function.

![[Excalidraw/actfn.excalidraw|150000]]

# Activation Functions


## I. Sigmoid   $$\sigma(x)=\frac{1}{1+e^{-x}}$$

```desmos-graph
y=1/(1+\exp(-x))
```

- Squashes numbers to range [0, 1]
- historically popular since they have nice interpretation as a saturating "firing rate" of a neuron

### Problems

### 1. Saturated neurons "kill" the gradients

![[Excalidraw/sigmoid.excalidraw]]

when x = -10 (very negative) then the output will be close to 0 meaning that the gradient becomes 0 and we are going to get very small gradient passing backwards.

when x=0 it will be in the regime or range 

when x=10 (large positive) the region is flat and it is going to kill the gradient


### 2. Sigmoid outputs are not zero - centered


![[Excalidraw/actfn.excalidraw|1000]]

Consider what happens when the input to a neuron is always positive, all are x's are going to be positive and they are gonna be multiplied to some weights 

then the updates are either all positive for the positive case or all negative for the negative case
this is why you also mean 0 mean data

![[Excalidraw/sigmoid2.excalidraw|150000]]

So we need to have zero meaned input so that we have positive and negative values both

### 3. exp() is a bit compute expensive

the exponential function is expensive not compared to all the calculations you are doing but a minor insight


## II. tanh(x)   $$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

```desmos-graph
y=\tanh(x)
```

- Squashes numbers to range [-1, 1]
- zero centered
- still kills gradients when saturated

it is better than sigmoid but still has its drawbacks


## III. ReLU (Rectified Linear Unit) f(x) = max(0, x)

- Does not saturate (in + region)
- Very computationally efficient
- Converges much faster than sigmoid/tanh in practice (e.g. 6x)
- Actually more biologically plausible

Note: if your input is negative then it is put to zero and if it is positive its going to pass through the identity

### Problems

### 1. Not zero - centered
### 2. Annoyance bcz if x is less than or equal to zero then the gradient will become zero and its only stable in its positive half

![[Excalidraw/relu.excalidraw|150000]]

the data cloud is our training data
if the weights are off the input then there will be dead relu
also if the learning rate is high then the weights are updated at a large difference and as a result the relu gets knocked off meaning after you start learning in between they go bad and then dead

It is a research problem but it does fine on the training data for now

people like to initialize the ReLU neurons with slightly positive bias like 0.01 to take care of the problem


## IV. Leaky ReLU $$\text{LeakyReLU}(x)=\begin{cases}x & x \ge 0\\0.01 x & x < 0\end{cases}$$

![[Excalidraw/leakyrelu.excalidraw|15000]]


- Does not saturate 
- Computationally efficient
- Converges much faster than sigmoid/tanh in practice
- will not die !!


## IV. Parametric ReLU $$\text{LeakyReLU}(x)=\begin{cases}x & x \ge 0\\\alpha x & x < 0\end{cases}$$
leaky relu is just a spl case of parametric relu where sigma is 0.01

### And then there is also ELU exponential Linear Units where the equation is as follows

$$\text{LeakyReLU}(x)=\begin{cases}x & x \ge 0\\\alpha (exp(x)-1) & x < 0\end{cases}$$
![[Excalidraw/exprelu.excalidraw]]

- all benefits of ReLU
- closer to zero mean outputs
- negative saturation regime compared to with Leaky ReLU adds some robustness to noise

but as it involves exp() it requires computation

# Maxout Neuron

# $$max(w_1^Tx+b_1, w_2^Tx+b_2)$$



- Does not have the basic form of dot product -----> non linearity
- Generalizes ReLU and Leaky ReLU 
- Linear Regime! Does not saturate! Does not die!

However it doubles the number of parameters/neurons

# Points to note and remember:-

1. Use ReLU. Be careful with your learning rates
2. Try out Leaky ReLU / Maxout / ELU
3. Try out tanh but dont expect much
4. Don't use sigmoid

---

# Data Preprocessing

![[Pasted image 20260604011114.png]]

we want to zero center the data as we dont want all of them to be positive or negative altogether
we should normalize the data corresponding to the std deviation to ensure all are in same range and contribute equally.

# Weight Initialization

![[Excalidraw/weightini.excalidraw|15000]]
2 layer neural network
### What happens when W=0 init is used ?
all the neurons might not be dead, they will all do the same thing and they are all going to output the same thing and have the same gradient and to say learning is zero here

### First idea : Small random numbers
(guassian with zero mean and 1e-2 std deviation)

```python
W = 0.01* np.random.randn(D, H)
```

in this case we are going to sample from a standard guassian but we are going to scale it so that the the std devn is 1e-2 (0.01)

Note: Works okay for small networks but it has problems with deeper networks

### Let us look at some activation statistics

E.g. 10-layer net with 500 neurons on each layer, using tanh non-linearities, and initializing as described in last slide.

```python
import numpy as np
import matplotlib.pyplot as plt

# assume some unit gaussian 500-D input data
D = np.random.randn(1000, 500)

hidden_layer_sizes = [500] * 10
nonlinearities = ['tanh'] * len(hidden_layer_sizes)

act = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}

Hs = {}

for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i - 1]  # input at this layer

    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]

    w = np.random.randn(fan_in, fan_out) * 0.01 # layer initialization

    H = np.dot(X, w)  # matrix multiply
    H = act[nonlinearities[i]](H)  # nonlinearity

    Hs[i] = H # cache result on this layer

# look at distribution at each layer
print('input layer had mean %f and std %f' % (np.mean(D), np.std(D)))

layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]

for i, H in Hs.items():
    print('hidden layer %d had mean %f and std %f'
          % (i + 1, layer_means[i], layer_stds[i]))

# ---------------------------------------------------------
# Plot means and standard deviations
# ---------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(list(Hs.keys()), layer_means, 'ob-')
plt.title('layer mean')
plt.xlabel('layer')
plt.ylabel('mean')

plt.subplot(122)
plt.plot(list(Hs.keys()), layer_stds, 'or-')
plt.title('layer std')
plt.xlabel('layer')
plt.ylabel('std')

plt.tight_layout()
plt.savefig("layer_mean_std.png")
plt.close()

print("Saved: layer_mean_std.png")

# ---------------------------------------------------------
# Plot raw distributions
# ---------------------------------------------------------

plt.figure(figsize=(20, 3))

for i, H in Hs.items():
    plt.subplot(1, len(Hs), i + 1)
    plt.hist(H.ravel(), 30, range=(-1, 1))
    plt.title(f'L{i+1}')

plt.tight_layout()
plt.savefig("layer_distributions.png")
plt.close()

print("Saved: layer_distributions.png")
```
![[layer_mean_std.png]]

![[layer_distributions.png]]


as we multiply the guassian graph in the first one shrinks down to zero by the tenth

Assuming this was our forward pass let us do out backward pass, what will the gradients look like
they will be collapsing to zero 

What if we try and solve this by making our weights big, lets sample this guassian with std devn 1 instead of 0.01

```python
import numpy as np
import matplotlib.pyplot as plt

# assume some unit gaussian 500-D input data
D = np.random.randn(1000, 500)

hidden_layer_sizes = [500] * 10
nonlinearities = ['tanh'] * len(hidden_layer_sizes)

act = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}

Hs = {}

for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i - 1]  # input at this layer

    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]

    w = np.random.randn(fan_in, fan_out) * 1 # layer initialization

    H = np.dot(X, w)  # matrix multiply
    H = act[nonlinearities[i]](H)  # nonlinearity

    Hs[i] = H # cache result on this layer

# look at distribution at each layer
print('input layer had mean %f and std %f' % (np.mean(D), np.std(D)))

layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]

for i, H in Hs.items():
    print('hidden layer %d had mean %f and std %f'
          % (i + 1, layer_means[i], layer_stds[i]))

# ---------------------------------------------------------
# Plot means and standard deviations
# ---------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(list(Hs.keys()), layer_means, 'ob-')
plt.title('layer mean')
plt.xlabel('layer')
plt.ylabel('mean')

plt.subplot(122)
plt.plot(list(Hs.keys()), layer_stds, 'or-')
plt.title('layer std')
plt.xlabel('layer')
plt.ylabel('std')

plt.tight_layout()
plt.savefig("layer_mean_std1.png")
plt.close()

print("Saved: layer_mean_std1.png")

# ---------------------------------------------------------
# Plot raw distributions
# ---------------------------------------------------------

plt.figure(figsize=(20, 3))

for i, H in Hs.items():
    plt.subplot(1, len(Hs), i + 1)
    plt.hist(H.ravel(), 30, range=(-1, 1))
    plt.title(f'L{i+1}')

plt.tight_layout()
plt.savefig("layer_distributions1.png")
plt.close()

print("Saved: layer_distributions1.png")
```
![[layer_mean_std1.png]]

![[layer_distributions1.png]]

here the weights are very big so we are always going to be in saturated regimes of too positive or too negative and what we are going to get her is exactly that. so all the gradients will be zero and our weights are not updated

One good rule of thumb is to use XAVIER INITIALIZATION


```python
import numpy as np
import matplotlib.pyplot as plt

# assume some unit gaussian 500-D input data
D = np.random.randn(1000, 500)

hidden_layer_sizes = [500] * 10
nonlinearities = ['tanh'] * len(hidden_layer_sizes)

act = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}

Hs = {}

for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i - 1]  # input at this layer

    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]

    w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in) # layer initialization

    H = np.dot(X, w)  # matrix multiply
    H = act[nonlinearities[i]](H)  # nonlinearity

    Hs[i] = H # cache result on this layer

# look at distribution at each layer
print('input layer had mean %f and std %f' % (np.mean(D), np.std(D)))

layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]

for i, H in Hs.items():
    print('hidden layer %d had mean %f and std %f'
          % (i + 1, layer_means[i], layer_stds[i]))

# ---------------------------------------------------------
# Plot means and standard deviations
# ---------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(list(Hs.keys()), layer_means, 'ob-')
plt.title('layer mean')
plt.xlabel('layer')
plt.ylabel('mean')

plt.subplot(122)
plt.plot(list(Hs.keys()), layer_stds, 'or-')
plt.title('layer std')
plt.xlabel('layer')
plt.ylabel('std')

plt.tight_layout()
plt.savefig("layer_mean_std2.png")
plt.close()

print("Saved: layer_mean_std2.png")

# ---------------------------------------------------------
# Plot raw distributions
# ---------------------------------------------------------

plt.figure(figsize=(20, 3))

for i, H in Hs.items():
    plt.subplot(1, len(Hs), i + 1)
    plt.hist(H.ravel(), 30, range=(-1, 1))
    plt.title(f'L{i+1}')

plt.tight_layout()
plt.savefig("layer_distributions2.png")
plt.close()

print("Saved: layer_distributions2.png")
```
![[layer_mean_std2.png]]

![[layer_distributions2.png]]

So what this formula is that, if we look at our W here, basically we specify that we want the variance of the input to be same as that of the output and then if you derive what the weights should be we get this formula

What this means is that if you have small number of inputs then we are going to divide by the smaller number and get larger weights and we want larger weights bcz with small inputs multiplied to weights you need larger weights to get the same variance and vice versa

One things this does assume is that there are linear activations meaning that we are in the active regions of tan h

Now if we use ReLU, as it is killing half of the units its actually halving the variance that you get out of this and distributions collapse.

```python
import numpy as np
import matplotlib.pyplot as plt

# assume some unit gaussian 500-D input data
D = np.random.randn(1000, 500)

hidden_layer_sizes = [500] * 10
nonlinearities = ['relu'] * len(hidden_layer_sizes)

act = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}

Hs = {}

for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i - 1]  # input at this layer

    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]

    w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in) # layer initialization

    H = np.dot(X, w)  # matrix multiply
    H = act[nonlinearities[i]](H)  # nonlinearity

    Hs[i] = H # cache result on this layer

# look at distribution at each layer
print('input layer had mean %f and std %f' % (np.mean(D), np.std(D)))

layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]

for i, H in Hs.items():
    print('hidden layer %d had mean %f and std %f'
          % (i + 1, layer_means[i], layer_stds[i]))

# ---------------------------------------------------------
# Plot means and standard deviations
# ---------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(list(Hs.keys()), layer_means, 'ob-')
plt.title('layer mean')
plt.xlabel('layer')
plt.ylabel('mean')

plt.subplot(122)
plt.plot(list(Hs.keys()), layer_stds, 'or-')
plt.title('layer std')
plt.xlabel('layer')
plt.ylabel('std')

plt.tight_layout()
plt.savefig("layer_mean_std3.png")
plt.close()

print("Saved: layer_mean_std3.png")

# ---------------------------------------------------------
# Plot raw distributions
# ---------------------------------------------------------

plt.figure(figsize=(20, 3))

for i, H in Hs.items():
    plt.subplot(1, len(Hs), i + 1)
    plt.hist(H.ravel(), 30, range=(-1, 1))
    plt.title(f'L{i+1}')

plt.tight_layout()
plt.savefig("layer_distributions3.png")
plt.close()

print("Saved: layer_distributions3.png")
```
![[layer_mean_std3.png]]

![[layer_distributions3.png]]

just like the ReLU graph the negative half is gone


The way to address this is to add divide by 2 while layer initialization and the intuition behind it is that, half the neurons get killed and the equivalent input has half these inputs.

```python
import numpy as np
import matplotlib.pyplot as plt

# assume some unit gaussian 500-D input data
D = np.random.randn(1000, 500)

hidden_layer_sizes = [500] * 10
nonlinearities = ['relu'] * len(hidden_layer_sizes)

act = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}

Hs = {}

for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i - 1]  # input at this layer

    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]

    w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2) # layer initialization

    H = np.dot(X, w)  # matrix multiply
    H = act[nonlinearities[i]](H)  # nonlinearity

    Hs[i] = H # cache result on this layer

# look at distribution at each layer
print('input layer had mean %f and std %f' % (np.mean(D), np.std(D)))

layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]

for i, H in Hs.items():
    print('hidden layer %d had mean %f and std %f'
          % (i + 1, layer_means[i], layer_stds[i]))

# ---------------------------------------------------------
# Plot means and standard deviations
# ---------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(list(Hs.keys()), layer_means, 'ob-')
plt.title('layer mean')
plt.xlabel('layer')
plt.ylabel('mean')

plt.subplot(122)
plt.plot(list(Hs.keys()), layer_stds, 'or-')
plt.title('layer std')
plt.xlabel('layer')
plt.ylabel('std')

plt.tight_layout()
plt.savefig("layer_mean_std4.png")
plt.close()

print("Saved: layer_mean_std4.png")

# ---------------------------------------------------------
# Plot raw distributions
# ---------------------------------------------------------

plt.figure(figsize=(20, 3))

for i, H in Hs.items():
    plt.subplot(1, len(Hs), i + 1)
    plt.hist(H.ravel(), 30, range=(-1, 1))
    plt.title(f'L{i+1}')

plt.tight_layout()
plt.savefig("layer_distributions4.png")
plt.close()

print("Saved: layer_distributions4.png")
```
![[layer_mean_std4.png]]

![[layer_distributions4.png]]

# Proper initialization is an active area of research

**Understanding the difficulty of training deep feedforward neural networks**
by Glorot and Bengio, 2010

**Exact solutions to the nonlineat dynamics of learning in deep linear neural networks**
by Saxe et al, 2013

**Random walk initialization for training very deep feedforward networks**
by Sussillo and Abbott, 2014

**Delving deep into rectifiers: Surpassing human-level performance on ImageNet Classification**
by He et al., 2015

**Data-dependent initializations of Convolutional Neural Networks**
by Krahenbuhl et al., 2015

**All you need is a good init**
by Mishkin and Matas, 2015

# Batch Normalization

the idea of wanting to keep activations in the gaussian range we want

you want unit gaussian activations? lets just make them so.

consider a batch of activations at some layer.

To make each dimension unit gaussian, apply:
# $$\hat{x}^{(k)} = \frac{x^{(x)} - E[x^{(k)}]} {\sqrt{Var[x^{(x)}]}} $$
## This is a vanilla differentiable function

here let us see what is happening

if we want to make it unit gaussian we can do this empirically, we can take the mean of the current batch and the variance and we can just normalize by this.

![[Excalidraw/batchnormal.excalidraw]]

N training examples in the current batch an then each batch has dimension D and we are going to compute variance and empirical mean independently for each dimension (for each feature element)

![[Excalidraw/BNplace.excalidraw]]

So one thing to note is that we are not certain that we always want unit gaussian inputs for our tanh, basically what we are saying by putting BN before tanh is that we dont want saturation.
BUT THE THING IS THAT WE SHOULD WANT TO HAVE A BIT OF SATURATION IN OUR HANDS BUT IN OUR CONTROL.

To address this problem we use something called squashing and scaling operation additionally to batch normalization.

What we are going to do here is that we are going to normalize the data and then scale it by a constant gamma and shift by another factor of beta.

# Normalize $$\hat{x}^{(k)} = \frac{x^{(x)} - E[x^{(k)}]} {\sqrt{Var[x^{(x)}]}} $$
And then allow the network to squash the range if it wants to:
# $$ y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$
Note, the network can learn:
$$ \gamma^{(k)} = \sqrt{Var[x^{(k)}]}$$
$$ \beta^{(k)} = E[x^{(k)}] $$
to recover the identity mapping.

In this case the network can learn your gamma to be your variance and beta to be mean.

## Batch Normalization

**Input:** Values of \(x\) over a mini-batch:

$$
B=\{x_1,\ldots,x_m\}
$$

**Parameters to be learned:**

$$
\gamma,\ \beta
$$

**Output:**

$$
\{y_i = BN_{\gamma,\beta}(x_i)\}
$$

### Mini-batch Mean

$$
\mu_B \leftarrow \frac{1}{m}\sum_{i=1}^{m}x_i
$$

### Mini-batch Variance

$$
\sigma_B^2 \leftarrow \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

### Normalize

$$
\hat{x}_i \leftarrow \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

### Scale and Shift

$$
y_i \leftarrow \gamma \hat{x}_i + \beta
= BN_{\gamma,\beta}(x_i)
$$

### Benefits

- Improves gradient flow through the network
- Allows higher learning rates
- Reduces the strong dependence on initialization
- Acts as a form of regularization
- Can slightly reduce the need for dropout
- more robust
- regularization in a way(all of the inputs tied together as a batch so the output is not solitary)

Note: at test time Batch Normalization layer functions differently:
The mean / std are not computed based on the batch. Instead, a single fixed empirical mean of activations during training is used.

(e.g. can be estimated during training with running averages)

# Babysitting the learning process

# step 1: Preprocess the data

![[Pasted image 20260604011114.png]]

# step 2: Choose the architecture

![[Excalidraw/weightini.excalidraw|15000]]

# step 3: double check that the loss is reasonable

first thing we do is that we initialize our network, we do a forward pass through it and we want to make sure that our loss is reasonable.

When our weights are small and we have a diffuse distribution then the softmax classifier loss is going to be negative log likelihood, which if we have 10 classes it will be something like negative log of one over 10, which here is around 2.3

```python
def init_two_layer_model(input_size, hidden_size, output_size):
	# initialize a model
	model = {}
	model['W1'] = 0.0001 * np.random.randn(input_size, hidden_size)
	model['b1'] = np.zeros(hidden_size)
	model['W2'] = 0.0001 * np.random.randn(hidden_size, output_size)
	model['b2'] = np.zeros(output_size)
	return model
```


```python
model = init_two_layer_model(32*32*3, 50, 10) # input size, hidden size, number of classes
loss, grad = two_layer_net(X_train, model, y_train, 1e3) # 1e3 ---> crank up regularization
print(loss)
```
expected outcome: 3.06859716482
loss went up, good (sanity check)

first we have to do at 0 regularization for 10 classes where we get 2.3 loss and when we crank up regularization the loss

```python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred



```

---
---

# CS231n Lecture 7 — Training Neural Networks Part 2

## Overview

This lecture focuses on:

1. Optimization Algorithms
2. Regularization Techniques
3. Transfer Learning

Goal:

```text
Reduce Training Error
+
Improve Generalization
```

---

# Optimization

## Vanilla Gradient Descent

Update rule:

$$
W \leftarrow W - \eta \nabla_W L
$$

Where:

- $W$ = parameters
- $\eta$ = learning rate
- $\nabla_W L$ = gradient of loss

Pseudo-code:

```python
while True:
    grad = compute_gradient(loss, W)
    W -= learning_rate * grad
```

---

## Problems with SGD

### 1. Poor Conditioning

Loss surface may look like a narrow valley.

```text
Steep in one direction
Flat in another
```

Result:

```text
Oscillation
+
Slow convergence
```

---

### 2. Saddle Points

A saddle point:

```text
Gradient = 0
Not a minimum
```

SGD may get stuck or move very slowly.

Important:

```text
In high-dimensional spaces,
saddle points are much more common
than local minima.
```

---

### 3. Noisy Gradients

Mini-batches only estimate the true gradient.

Instead of:

$$
\nabla L
$$

we get:

$$
\nabla \hat L
$$

which contains noise.

Result:

```text
Jittery updates
Slow convergence
```

---

# SGD with Momentum

## Idea

Instead of following only the current gradient:

```text
Remember previous gradients.
```

Build velocity.

---

## Update Rule

Velocity:

$$
v_{t+1} = \rho v_t - \eta \nabla L(W_t)
$$

Parameter update:

$$
W_{t+1}=W_t+v_{t+1}
$$

---

## Intuition

Imagine a ball rolling downhill.

```text
Gradient
→ direction

Momentum
→ speed
```

Benefits:

- Faster convergence
- Reduces oscillation
- Escapes shallow minima
- Handles noisy gradients

---

## Typical Values

```python
momentum = 0.9
```

or

```python
momentum = 0.99
```

---

# Nesterov Momentum

## Motivation

Momentum can overshoot.

Nesterov:

```text
Look ahead first
Then compute gradient
```

---

## Update

Compute gradient at the future position.

$$
v_{t+1}
=
\rho v_t
-
\eta \nabla f(W_t+\rho v_t)
$$

---

## Benefit

More informed update.

Usually:

```text
Slightly better than momentum
```

---

# AdaGrad

## Idea

Use different learning rates for different parameters.

Frequently updated parameters:

```text
Smaller learning rate
```

Rarely updated parameters:

```text
Larger learning rate
```

---

## Update

Accumulate squared gradients:

$$
G_t = G_{t-1}+g_t^2
$$

Update:

$$
W_t
=
W_{t-1}
-
\frac{\eta g_t}
{\sqrt{G_t}+\epsilon}
$$

---

## Problem

Accumulated gradients keep growing.

Therefore:

$$
\sqrt{G_t}
\rightarrow \infty
$$

Learning rate:

$$
\rightarrow 0
$$

Eventually learning stops.

---

# RMSProp

## Idea

Fix AdaGrad.

Instead of storing all history:

```text
Use moving average
of squared gradients.
```

---

## Update

$$
G_t
=
\rho G_{t-1}
+
(1-\rho)g_t^2
$$

Parameter update:

$$
W_t
=
W_{t-1}
-
\frac{\eta g_t}
{\sqrt{G_t}+\epsilon}
$$

---

## Benefits

- Adaptive learning rates
- Doesn't decay forever
- Works well in practice

---

# Adam

## Most Popular Optimizer

Adam combines:

```text
Momentum
+
RMSProp
```

---

## First Moment

Momentum estimate:

$$
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
$$

---

## Second Moment

Squared gradient estimate:

$$
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
$$

---

## Bias Correction

Because estimates start at zero:

$$
\hat m_t
=
\frac{m_t}
{1-\beta_1^t}
$$

$$
\hat v_t
=
\frac{v_t}
{1-\beta_2^t}
$$

---

## Final Update

$$
W_t
=
W_{t-1}
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
$$

---

## Stanford Recommendation

Good default:

```python
beta1 = 0.9
beta2 = 0.999
lr = 1e-3
```

or

```python
lr = 5e-4
```

A great starting point for most models. :contentReference[oaicite:1]{index=1}

---

# Learning Rate Decay

## Why?

Large learning rate:

```text
Fast progress
```

Near optimum:

```text
Need smaller steps
```

---

## Methods

### Step Decay

```text
Reduce LR every few epochs
```

Example:

```python
lr *= 0.5
```

---

### Exponential Decay

$$
lr_t
=
lr_0 e^{-kt}
$$

---

### 1/t Decay

$$
lr_t
=
\frac{lr_0}{1+t}
$$

---

## Practical Advice

More important for:

```text
SGD
Momentum
```

Less critical for:

```text
Adam
```

---

# First vs Second Order Optimization

## First Order

Uses:

$$
\nabla L
$$

Examples:

- SGD
- Momentum
- RMSProp
- Adam

---

## Second Order

Uses:

$$
\nabla L
+
H
$$

where:

$$
H
=
\text{Hessian Matrix}
$$

---

## Newton's Method

Update:

$$
W
=
W-H^{-1}\nabla L
$$

Advantage:

```text
No learning rate
```

---

## Why Not Use It?

Deep networks have:

```text
Millions of parameters
```

Hessian size:

$$
O(N^2)
$$

Inversion:

$$
O(N^3)
$$

Too expensive.

---

## L-BFGS

Approximate second-order method.

Works well:

```text
Full batch
Deterministic optimization
```

Not ideal:

```text
Mini-batch training
```

---

## Practical Recommendation

Stanford:

```text
Adam is usually the best default.
```

If full-batch training is possible:

```text
Try L-BFGS.
```

---

# Model Ensembles

## Idea

Train multiple models.

At test time:

```text
Average predictions.
```

---

## Benefit

Typically:

```text
~2% performance gain
```

---

## Snapshot Ensembles

Instead of training many networks:

```text
Save checkpoints
from one network.
```

Use them as ensemble members.

---

## Polyak Averaging

Keep running average of weights:

$$
W_{avg}
=
\frac1T
\sum_t W_t
$$

Use averaged weights at test time.

---

# Regularization

Goal:

```text
Reduce Overfitting
```

---

## L2 Regularization

Weight decay.

Loss:

$$
L
=
L_{data}
+
\lambda \sum W^2
$$

Effect:

```text
Encourages small weights
```

---

## L1 Regularization

Loss:

$$
L
=
L_{data}
+
\lambda \sum |W|
$$

Effect:

```text
Sparse weights
```

---

## Elastic Net

Combination:

$$
L1 + L2
$$

---

# Dropout

## Idea

Randomly remove neurons during training.

---

## Example

Dropout rate:

```python
p = 0.5
```

Each neuron:

```text
50% chance active
50% chance removed
```

---

## Why It Works

Prevents:

```text
Co-adaptation
```

Network cannot rely on specific neurons.

Learns redundant robust features.

---

## Ensemble Interpretation

Every dropout mask:

```text
A different network
```

Training performs implicit model averaging.

---

## Test Time

All neurons active.

Need scaling:

$$
output_{test}
=
E[output_{train}]
$$

---

## Inverted Dropout

Modern implementation.

Scale during training.

Result:

```text
No scaling needed at test time.
```

---

# Data Augmentation

## Goal

Artificially increase dataset size.

---

## Horizontal Flips

```text
Cat → Mirrored Cat
```

Label unchanged.

---

## Random Crops

Randomly sample image regions.

Improves robustness.

---

## Random Scaling

Resize image before cropping.

---

## Color Jitter

Random:

- Brightness
- Contrast
- Color

---

## Advanced Augmentation

- Rotation
- Translation
- Shearing
- Stretching
- Lens Distortion

Stanford advice:

```text
Get creative.
```

---

# Common Regularization Pattern

Training:

```text
Add randomness
```

Testing:

```text
Average out randomness
```

Examples:

- Dropout
- BatchNorm
- Data Augmentation

---

# Transfer Learning

## Motivation

Training CNNs from scratch:

```text
Needs lots of data
```

Transfer learning solves this.

---

## Step 1

Train on large dataset.

Example:

```text
ImageNet
```

---

## Step 2

Reuse learned features.

Replace final classifier.

---

## Small Dataset

Freeze:

```text
Convolution layers
```

Train:

```text
New classifier layer
```

---

## Larger Dataset

Fine-tune:

```text
Last few layers
```

---

## Very Large Dataset

Fine-tune:

```text
Many layers
```

Use lower learning rate:

```python
new_lr = old_lr / 10
```

---

# Practical Transfer Learning Rules

| Dataset Similarity | Dataset Size | Strategy |
|----------|----------|----------|
| Similar | Small | Linear classifier on top |
| Similar | Large | Fine-tune few layers |
| Different | Small | Difficult problem |
| Different | Large | Fine-tune many layers |

---

# Why Transfer Learning Matters

Transfer learning is the default approach today.

Applications:

- Object Detection
- Image Captioning
- Medical Imaging
- Autonomous Driving
- Remote Sensing

---

# Final Takeaways

## Optimization

Know:

- SGD
- Momentum
- Nesterov
- AdaGrad
- RMSProp
- Adam

Adam is the best default.

---

## Regularization

Know:

- L1
- L2
- Dropout
- Data Augmentation

---

## Transfer Learning

Know:

```text
Pretrain
↓
Freeze
↓
Fine-tune
```

This is the most important practical technique in modern deep learning.

