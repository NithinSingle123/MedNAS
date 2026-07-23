
# Computational Graphs

here we are gonna talk about how to calculate the analytic gradient for the arbitrarily complex functions, using a framework.

![[computational_graphs.excalidraw]]

We can use this kind of graphs to represent any function where the nodes of the graph are steps of computation we go through.

the above graph is the example of a linear classifier

- here the inputs are x and w
- and then this multiplication node represents the matrix multiplication of x with w outputting the vector of scores
- hinge loss computing data loss term Li
- regularization term R(W)
- and the total loss L

The advantage is that once we can express a function using a computational graph then we can use a technique back propagation which is going to recursively use the chain rule in order to compute the gradient with respect to every variable in the computational graph.


## How does back propagation work ?

A simple example 
f(x, y, z) = (x + y)z
eg. x = -2, y = 5, z = -4

![[backpropeg1.excalidraw]]


$$
q = x + y
\qquad
\frac{\partial q}{\partial x}=1,
\quad
\frac{\partial q}{\partial y}=1
$$

$$
f = qz
\qquad
\frac{\partial f}{\partial q}=z,
\quad
\frac{\partial f}{\partial z}=q
$$


$$
\text{Want:}
\qquad
\frac{\partial f}{\partial x},
\quad
\frac{\partial f}{\partial y},
\quad
\frac{\partial f}{\partial z}
$$

So here we are going to start at the very end of the computational graph and then we are going to work backwards and compute all the gradients along the way.

the red digits in the graph are the gradients and below is written how we calculated it....

so the gradient near f is calculated by finding partial with respect to previous variable, for f it is just $$\frac{\partial f}{\partial f} = 1$$
for the z term gradient in a similar fashion we find out gradient by calculating the below expression $$\frac{\partial f}{\partial z}=q=3$$
for the q term gradient it is $$\frac{\partial f}{\partial q}=z=-4$$
for the x and y we have to still find the gradients using the equations
$$\frac{\partial f}{\partial x}$$ $$\frac{\partial f}{\partial y}$$
but as the x and y are not directly connected to f and are connected through an intermediary node q the above gradients can be found by using the chain rule as follows:

$$\frac{\partial f}{\partial x}=\frac{\partial f}{\partial q} \frac{\partial q}{\partial x}$$
$$\frac{\partial f}{\partial y}=\frac{\partial f}{\partial q} \frac{\partial q}{\partial y}$$
![[backpropoutline.excalidraw]]

Now we will look at a more complex example
$$
f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}}
$$


some relevant formulas useful forward are below
$$
f(x)=e^x
\qquad \rightarrow \qquad
\frac{df}{dx}=e^x
$$

$$
f_a(x)=ax
\qquad \rightarrow \qquad
\frac{df}{dx}=a
$$

$$
f(x)=\frac{1}{x}
\qquad \rightarrow \qquad
\frac{df}{dx}=-\frac{1}{x^2}
$$

$$
f_c(x)=c+x
\qquad \rightarrow \qquad
\frac{df}{dx}=1
$$

![[Excalidraw/backpropprob1.excalidraw|12000]]

the sigmoid analogy just tells us that you can group how many ever nodes to simplify the computation graph using predefined function as long as you can find the local gradient

## add gate : gradient distributor
## max gate : gradient router
## multiplication gate : gradient switcher

![[Excalidraw/gradientaddrule.excalidraw]]

Gradients add at the branches

What happens when we have vectors, so instead of x, y and z being scalars, they are vectors.

In this case what happens is that the flow remains the same but the only thing is that now our gradients are going to be JACOBIAN MATRICES

![[backpropoutline.excalidraw]]

![[Excalidraw/vectorizedopns.excalidraw]]

common size you might in cnn s later on

The size of the Jacobian Matrix 4096 squared and it is pretty large. And In practice this is going to be even large and this is because we are going to process an entire minibatch (e.g. 100) of examples at one time:

<mark style="background: #FFF3A3A6;">and that gives us a matrix of 409600 squared</mark>

So in practice we dont need to compute this huge jacobian


## A Vectorized Example: $$
f(x,W)=\|W\cdot x\|^2
=
\sum_{i=1}^{n}(W\cdot x)_i^2
$$

![[Excalidraw/vectoreg.excalidraw|12000]]

## Modularized implementation of forward and backward pass

```python
class ComputationalGraph(object):
	#...
	def forward(inputs):
		# 1. [pass inputs to input gates..]
		# 2. forward the computational graph:
		for gate in self.graph.nodes_topologically_sorted():
			gate.forward()
		return loss # the final gate in the graph outputs the loss
	def backward():
		for gate in reversed(self.grah.nodes_topologically_sorted()):
			gate.backward() # little piece of backprop (chain rule applied)
		return inputs_gradients
```

In forward pass we calculate the loss and then in the backward pass calculates the gradients

## Example for scalars

![[Excalidraw/passcode.excalidraw]]

```python
class Multiplygate(Object):
	def forward(x, y):
		z = x*y
		self.x = x # must keep these around!
		self.y = y
		return z
	def backward(dz):
		dx = self.y * dz # [dL/dz * dz/dx]
		dy = self.x * dz # [dL/dz * dz/dy]
		return [dx, dy]
```



# Neural Networks

Simpler functions stacked on top of each other, and they are stacked in a hierarchical way to make up a more complex non linear function

<mark style="background: #FFF3A3A6;">Note: - if you dont put non linearities in between linearities then the stack will collapse to a linear function altogether
</mark>

## full implementation of training a 2-layer Neural Network needs ~20 lines:

```python
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, W), randn(H, D_out)

for t in range(2000):
	h = 1 / (1+np.exp(=x.dot(w1)))
	y_pred = h.dot(w2)
	loss = np.sqaure(y_pred - y).sum()
	print(t, loss)
	
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h.T.dot(grad_y_pred)
	grad_h = grad_y_pred.dot(w2.T)
	grad_w1 = x.T.dot(grad_h * h * (1-h))
	
	w1 -= le-4 * grad_w1
	w2 -= le-4 * grad_w2
```


## biological analogy to your brain

