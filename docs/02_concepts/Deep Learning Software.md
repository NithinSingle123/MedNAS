## The point of deep learning frameworks instead of writing your own code

(1) Easily build big computational graphs
(2) Easily compute gradients in computational graphs
(3) Run it all efficiently on GPU (wrap cuDNN, cuBLAS, etc)

# An Example Numpy Computational Graph

```python
import numpy as np
np.random.seed(0)

N, D = 3, 4

x = np.random.randn(N, D)
y = np.random.randn(N, D)
z = np.random.randn(N, D)

a = x * y
b = a + z
c = np.sum(b)
```

![[Excalidraw/numpycompgraph.excalidraw]]
Suppose we want to find the gradient of c with respect of x, y and z, in Numpy, you have to write the backward pass yourself and it becomes a pain in the ass for bigger models

Now look at the tensorflow version of this,....
```python
import numpy as np
np.random.seed(0)
import tensorflow as tf

N, D = 3, 4

with th.device('/cpu:0'): # with th.device('/gpu:0'): ---> for using gpu
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	z = tf.placeholder(tf.float32)

	a = x * y
	b = a + z
	c = tf.reduce_sum(b)

grad_x, grad_y, grad_z = tf.gradients(C, [x, y, z])

with tf.session as sess:
	values = {
		x: np.random.randn(N, D),
		y: np.random.randn(N, D),
		z: np.random.randn(N, D),
	}
	out = sess.run([c, grad_x, grad_y, grad_z], feed_dict = values)
	c_val, grad_x_val, grad_y_val, g: ad x val = out
```

Now look at the pytorch version of this...
```python
import torch
from torch.autograd import Variable

N, D = 3, 4

x = Variable(torch.randn(N, D) '''.cuda (for gpu)''', requires_grad=True)
y = Variable(torch.randn(N, D), requires_grad=True)
z = Variable(torch.randn(N, D), requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
```

# TensorFlow: training a 2 layer fully connected ReLU network on random data with L2 euclidean loss on random data
```python
import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100

# as the name suggests they are placeholders
x = tf.placeholder(tf.float32, shape = (N, D))
y = tf.placeholder(tf.float32, shape = (N, D))
w1 = tf.placeholder(tf.float32, shape = (D, H))
w2 = tf.placeholder(tf.float32, shape = (H, D))

# forward pass ---> still just building graph, no computation here!
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis = 1))

# ask tensorflow to compute gradient of loss with respect to w1 and w2 --> again no computation
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
# here we are done building the graph 

# so we enter the session and start feeding the data
with tf.Session() as sess:
# here just creating concrete actual values using numpy and storing in dictionaries

# feed in the numpy arrays and get arrays for loss, grad_w1 and grad_w2
	values = {x: np.random.randn(N, D),
			  w1: np.random.randn(D, H),
			  w2: np.random.randn(H, D),
			  y: np.random.randn(N, D),}
	out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
	loss_val, grad_w1_val, grad_w2_val = out
	
# HERE IT TAKES ONLY A FEW MORE LINES TO TRAIN THE NETWORK

```

In tensorflow you divide the computation into two major stages:
1. DEFINE THE COMPUTATIONAL GRAPH
2. RUN THE GRAPH MANY TIMES (reuse it many times)

**For training the network it only takes a few extra lines**
```python
with tf.Session() as sess:
	values = {x: np.random.randn(N, D),
			  w1: np.random.randn(D, H),
			  w2: np.random.randn(H, D),
			  y: np.random.randn(N, D),}
	learning_rate = 1e-5

# training the network
	for t in range(50):
		out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
		loss_val, grad_w1_val, grad_w2_val = out
		values[w1] -= learning_rate * grad_w1_val
		values[w2] -= learning_rate * grad_w2_val
```

But there is a prob here!
Every time we execute the graph, we are feeding the weights into the graph directly and each time we run it computes gradients and then it is copied into arrays and then into the next iteration. But what this does it makes this process computationally very expensive because of the bottleneck that is created in the process of copying info from GPU to CPU and vice versa.
*WE NEED TO FIX THAT*

The fix to this is that we change w1 and w2 to be variables instead of placeholders and this is because it lives inside the computational graph and it persists for the multiple iterations of the graph. 

```python
N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape = (N, D))
y = tf.placeholder(tf.float32, shape = (N, D))
w1 = tf.variable(tf.random_normal, shape = (D, H)) #tf.random_normal tells tf how to initialize
w2 = tf.variable(tf.random_normal, shape = (H, D))

h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis = 1))

learning_rate = 1e-5
new_w1 = w1.assign(w1 - learning_rate * grad_w1) # instead of passing weights around as numpy arrays we do this job inside the graph itself
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) 
	values = {x: np.random.randn(N, D),
			  y: np.random.randn(N, D),}
	for t in range(50):
		loss_val, = sess.run([loss], feed_dict = values)
```
But the problem here is that if you run this code you will find that the loss is not going down and the assign calls are not actually being executed.

The solution in this is to tell tf explicitly that you need to update the new_w1 and new_w2 for the next iteration and this happens because tensorflow is a slick bastard.
What happens here is that it only computes the lines which are need to find the loss and that would be the last two lines.

so we add a new line before session start:
```python
updates = tf.group(new_w1, new_w2)

# now this will be reflected in the last line of the code as such
for t in range(50):
		loss_val, = sess.run([loss, updates], feed_dict = values)
```
But now we are back to square 0 where in the problem of data transfer between CPU and GPU at every iteration arises again..

There is a little trick to this instead..
**You kind of add a dummy node to the graph with fake data dependencies and we just say that this dummy node updates has these data dependencies of new_w1 and new_w2.

## TensorFlow: Optimizer

Now here is the operation that solves our problems
so how to use this is that you write these lines before getting into a session...
```python
optimizer = tf.train.GradientDescentOptimizer(1e-5)
updates = optimizer.minimize(loss)
# and we put updates in the last line as usual
```

Here what this "optimizer" keyword does is that it has all the different optimizers, so instead of hassling over manually writing the formulas it does everything by itself and the best part about it is the next line where in optimizer knows that it has to update the values to the next iteration just by adding a line and mentioning it in the last line the update is done by optimizer itself and the implementation is the same as the intuition we did with tf.group

There are some predefined keywords that simplify the job even more 
For example, 
```python
loss = tf.losses.mean_sqaured_error(y_pred, y)
# this one line helps us avoid manual hassle and get L2 loss directly
```

And because of this manual labor bullshit it will become extremely hard to follow when we do the same manual bullshit for higher levels of cnns and stuff like that.
So what we are going to do is use **MAGIC LINES** 

```python
N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape = (N, D))
y = tf.placeholder(tf.float32, shape = (N, D))

# use xavier initializer
init = tf.contrib.layers.xavier_initializer()

# tf.layers automatically sets up weight and bias for us
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

loss = tf.losses.mean_squared_error(y_pred, y)

optimizer = tf.train.GradientDescentOptimizer(1e0)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) 
	values = {x: np.random.randn(N, D),
			  y: np.random.randn(N, D),}
	for t in range(50):
		loss_val, = sess.run([loss, updates], feed_dict = values)
```

# Keras: High-Level Wrapper
Keras is a layer on top of TensorFlow, makes common things easy to do (also supports theano backend)

```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

N, D, H = 64, 1000, 100

# defined model object as a sequence of layers
model = Sequential()
model.add(Dense(input_dim=D, output_dim=H))
model.add(Activation('relu'))
model.add(Dense(input_dim=H, output_dim=D))

# defined optimizer object
optimizer = SGD(lr=1e0)
model.compile(loss='mean_squared_error', optimizer=optimizer)

x = np.random.randn(N, D)
y = np.random.randn(N, D)

# train the whole model with a single line
history = model.fit(x, y, nb_epoch=50, batch_size=N, verbose=0)
```

# TensorFlow: Tensorboard
Add logging to code to record loss, stats, etc Run server and get pretty graphs
KIND OF LIKE A DASHBOARD

# TensorFlow: Distributed Version
tf also allows you to split one graph and run on multiple machines

# PyTorch (Facebook)

**PyTorch: Three Levels of Abstraction**
1. Tensor: Imperative ndarray, but runs on GPU
2. Variable: Node in a computational graph; stores data and gradient
3. Module: A neural network layer; may store state or learnable weights

## PyTorch Tensors

- PyTorch Tensors are just like numpy arrays, but they can run on GPU.
- No built-in notion of computational graph, or gradients, or deep learning

below is the code for a two-layer net using Pytorch Tensors:
```python
import torch

dtype = torch.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
	
```


