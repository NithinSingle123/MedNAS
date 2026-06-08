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

# dtype = torch.FloatTensor
# to run on GPU, just cast tensors to a cuda datatype
d_type = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)
	loss = (y_pred - y).pow(2).sum
	
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)
	
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2
```

## Pytorch: Autograd

- A PyTorch Variable is a node in a computational graph
- x.data is a Tensor
- x.grad is a Variable of gradients (same shape as x.data)
- x.grad.data is a Tensor of gradients

```python
import torch
from torch.autograd import Variable

N, D, H = 64, 1000, 100
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
	# fwd pass looks exactly same as tensors version but everything is a variable now
	relu = ReLU() # self defined function, mentioned below
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	loss = (y_pred - y).pow(2).sum()
	
	# Compute gradient of loss with respect to w1 and w2 (zero out grads first)
	if w1.grad: w1.grad.data.zero_()
	if w2.grad: w2.grad.data.zero_()
	loss.backward()
	
	# make gradient step on weights
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data
```

- PyTorch Tensors and Variables have the same API, meaning any code that worked on PyTorch tensors you can just cast them as variable except now you are building up a computational graph, rather than just imperative operations

## PyTorch: New Autograd Functions

```python
class ReLU(torch.autograd.Function):
	def forward(self, x):
		self.save_for_backward(x)
		return x.clamp(min=0)
		
	def backward(self, grad_y):
		x, = self.saved_tensors
		grad_input = grad_y.vlone()
		grad_input[x < 0] = 0
		return grad_input
```

- Define your own autograd functions by writing forward and backward for Tensors (similar to modular layers in A2) and implement it in the main code.

## PyTorch: nn

- Higher-level wrapper for working with neural nets
- Similar to keras

```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# define our model as a sequence of layers
model = torch.nn.Sequential(
			torch.nn.Linear(D_in, H),
			torch.nn.ReLU(),
			torch.nn.Linear(H, D_out))

# nn also defines common loss functions
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
	# fwd pass: feed data to model, and prediction to the loss function
	y_pred = model(x)
	loss = loss_fn(y_pred, y)
	
	# bwd pass: compute all grads
	model.zero_grad()
	loss.backward()
	
	# make gradient step on each model parameter
	for param in model.parameters():
		param.data -= learning_rate * param.grad.data
```

## PyTorch: optim

```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
			torch.nn.Linear(D_in, H),
			torch.nn.ReLU(),
			torch.nn.Linear(H, D_out))

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

# Use an optimizer for different update rules
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
	y_pred = model(x)
	loss = loss_fn(y_pred, y)
	
	model.zero_grad()
	loss.backward()
	
	optimizer.step()
```

## PyTorch: nn (Define new Modules)

- A PyTorch Module is a neural net layer; it inputs and outputs Variables
- Modules can contain weights (as Variables) or other Modules
- You can define your own Modules using autograd

```python
import torch
from torch.autograd import Variable

# Define our whole model as a single module
class TwoLayerNet(torch.nn.Module):
	# initializer sets up two children (Modules can contain modules)
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

	# define fwd pass using child modules and autograd ops on Variables
	# No need to define backward - autograd will handle it
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct and train an instance of our model
model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- Here you can write your own class which defines your entire model as a single new nn module class.
- A module is just kind of a neural network layer that can contain either other modules or trainable weights or other kinds of states.

## PyTorch: DataLoaders

- A DataLoader wraps a Dataset and provides minibatching, shuffling multithreading, for you
- When you need to load custom data, just write your own Dataset class

```python
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loader = DataLoader(TensorDataset(x, y), batch_size=8)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# iterate over the loader to form minibatches
# Loader gives tensors so you need to wrap in Variables
for epoch in range(10):
    for x_batch, y_batch in loader:
        x_var, y_var = Variable(x_batch), Variable(y_batch)

        y_pred = model(x_var)
        loss = criterion(y_pred, y_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## PyTorch: Pretrained Models

Super easy to use pertained models with torchvision
https://github.com/pytorch/vision

```python
import torch
import torchvision

alexnet = torchvision.models.alexnet(pretarined=True)
vgg16 = torchvision.models.vgg16(pretrained=True)
resnet101 = torchvision.models.resnet101(pretrained=True)
```

## PyTorch: Visdom

Somewhat similar to TensorBoard: add logging to your code, then visualized in a browser
(lets you visualize computational graph)

## Static vs Dynamic Graphs

- TensorFlow: Build graph once, then run many times each (static)
- PyTorch: Each forward pass defines a new graph (dynamic)

# Static vs dynamic: Optimization

- With static graphs, framework can optimize the graph for you before it runs!

![[Excalidraw/dynvsstat.excalidraw|15000]]

## Static vs Dynamic: Searialization

- Static: Once the graph is built, can serialize it (save it in disk) and run it without the code that built the graph
- Dynamic: Graph building and execution are intertwined, so always need to keep code around

## Static vs Dynamic: Conditional

# $$ y =  \begin{cases}  w_1 x & \text{if } z > 0 \\  w_2 x & \text{otherwise}  \end{cases}$$
PyTorch (Dynamic Graph)
```python
from torch.autograd import Variable
import torch

N, D, H = 3, 4, 5

x = Variable(torch.randn(N, D))
w1 = Variable(torch.randn(D, H))
w2 = Variable(torch.randn(D, H))

z = 10

if z > 0:
    y = x.mm(w1)
else:
    y = x.mm(w2)
```

Key idea:
- PyTorch executes normal Python.
- The if statement is evaluated immediately.
- Only the executed branch becomes part of the computation graph.

TensorFlow (Static Graph)
```python
import tensorflow as tf
import numpy as np

N, D, H = 3, 4, 5

x = tf.placeholder(tf.float32, shape=(N, D))
z = tf.placeholder(tf.float32, shape=None)
w1 = tf.placeholder(tf.float32, shape=(D, H))
w2 = tf.placeholder(tf.float32, shape=(D, H))


def f1():
    return tf.matmul(x, w1)


def f2():
    return tf.matmul(x, w2)


y = tf.cond(tf.less(z, 0), f1, f2)

with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        z: 10,
        w1: np.random.randn(D, H),
        w2: np.random.randn(D, H),
    }

    y_val = sess.run(y, feed_dict=values)
```

Key idea:
- TensorFlow builds a graph first.
- The condition cannot be handled using a normal Python if.
- Instead, special graph operators such as tf.cond() must be inserted into the graph.

## Static vs Dynamic: Loops

# $$y_t = (y_{t-1} + x_t) * w$$

![[Pasted image 20260608163016.png]]

PyTorch (Dynamic Graph)
```python
from torch.autograd import Variable
import torch

T, D = 3, 4

y0 = Variable(torch.randn(D))
x = Variable(torch.randn(T, D))
w = Variable(torch.randn(D))

y = [y0]

for t in range(T):
    prev_y = y[-1]
    next_y = (prev_y + x[t]) * w
    y.append(next_y)
```

Key idea:
- PyTorch uses a normal Python loop.
- The computation graph is built dynamically as each iteration executes.

TensorFlow (Static Graph)
```python
import tensorflow as tf
import numpy as np

T, N, D = 3, 4, 5

x = tf.placeholder(tf.float32, shape=(T, D))
y0 = tf.placeholder(tf.float32, shape=(D,))
w = tf.placeholder(tf.float32, shape=(D,))


def f(prev_y, cur_x):
    return (prev_y + cur_x) * w


y = tf.foldl(f, x, y0)

with tf.Session() as sess:
    values = {
        x: np.random.randn(T, D),
        y0: np.random.randn(D),
        w: np.random.randn(D),
    }

    y_val = sess.run(y, feed_dict=values)
```

Key idea:
- TensorFlow cannot use a normal Python loop inside the graph.
- Special graph operators such as tf.foldl() must be used to represent looping behavior.

# Dynamic Graphs in TensorFlow

- TensorFlow Fold makes dynamic graphs easier in TensorFlow through dynamic batching

**Why should we care about dynamic graphs in general ?**
- So one option is Recurrent networks
![[Pasted image 20260608163439.png]]

**Dynamic Graph Applications**
- Recurrent networks
- Recursive networks
![[Pasted image 20260608163643.png]]

# CS231n Lecture 8 — Slides 135-152
# Caffe, Caffe2, and Framework Advice

---

# Caffe (UC Berkeley)

## Caffe Overview

- Core written in C++
- Has Python and MATLAB bindings
- Good for training or finetuning feedforward classification models
- Often no need to write code
- Not used as much in research anymore
- Still popular for deploying models

---

# Caffe Training / Finetuning Workflow

No need to write code!

1. Convert data
   - Run a script

2. Define network
   - Edit prototxt

3. Define solver
   - Edit prototxt

4. Train
   - Run a script
   - Optionally use pretrained weights

---

# Step 1: Convert Data

### Preferred Method

- DataLayer reading from LMDB

Create LMDB using:

```bash
convert_imageset
```

Need a text file where each line contains:

```text
[path/to/image.jpg] [label]
```

Alternative:

- Create HDF5 manually using h5py

---

### Other Data Input Methods

- ImageDataLayer
  - Read directly from image files

- WindowDataLayer
  - Object detection

- HDF5Layer
  - Read HDF5 files

- Python Interface
  - Read from memory

Most are harder to use
(except Python interface)

---

# Step 2: Define Network (prototxt)

Network architecture is defined in:

```text
.prototxt
```

Example:

ResNet-152 deploy prototxt

Issues:

- Prototxt becomes ugly for large models
- ResNet-152 prototxt is 6775 lines long
- Not compositional
- Cannot define a residual block once and reuse it easily

---

# Step 3: Define Solver (prototxt)

Create a file defining:

```text
SolverParameter
```

For finetuning:

- Copy an existing solver.prototxt
- Change:

```text
net
snapshot_prefix
base_learning_rate
max_iter
snapshot
```

Recommended:

- Divide learning rate by 100

---

# Step 4: Train

Example:

```bash
./build/tools/caffe train \
-gpu 0 \
-model path/to/trainval.prototxt \
-solver path/to/solver.prototxt \
-weights path/to/pretrained_weights.caffemodel
```

GPU options:

```bash
-gpu -1
```

CPU only

```bash
-gpu all
```

Multi-GPU training

---

# Caffe Model Zoo

Includes:

- AlexNet
- VGG
- GoogLeNet
- ResNet
- Many others

---

# Caffe Python Interface

Documentation is limited.

Important files:

### _caffe.cpp

Exports:

- Blob
- Layer
- Net
- Solver

### pycaffe.py

Adds additional helper methods to Net

---

# What is the Python Interface Good For?

- Interfacing with NumPy
- Feature extraction
- Running forward passes
- Running backward passes
- DeepDream-style gradient visualization
- Creating Python layers with NumPy
  - CPU only

---

# Caffe Pros

- Good for feedforward networks
- Good for finetuning
- Can train models without writing code
- Useful Python interface
- Can deploy without Python

---

# Caffe Cons

- Need C++ / CUDA for new GPU layers
- Poor support for recurrent networks
- Large networks become cumbersome
  - GoogLeNet
  - ResNet

---

# Caffe2 (Facebook)

## Caffe2 Overview

- Released very recently (at lecture time)
- Uses static computation graphs
- Similar philosophy to TensorFlow
- Core written in C++
- Nice Python interface
- Train in Python
- Serialize model
- Deploy without Python
- Works on:
  - iOS
  - Android
  - Other platforms

---

# Industry Direction (2017)

Google:

TensorFlow

Facebook:

PyTorch + Caffe2

Research -------------------- Production

Goal:

"One framework to rule them all"

---

# Justin Johnson's Advice (2017)

### TensorFlow

Use when:

- General projects
- Large community support
- Broad adoption
- Multi-machine computation

Consider pairing with:

- Keras
- Sonnet
- Other wrappers

---

### PyTorch

Best choice for research.

Pros:

- Flexible
- Dynamic graphs
- Research-friendly

Cons:

- Newer framework
- Occasional rough edges

---

### TensorFlow

Use when:

- One graph across many machines
- Large-scale distributed training

---

### Production Deployment

Consider:

- TensorFlow
- Caffe
- Caffe2

---

### Mobile Deployment

Consider:

- TensorFlow
- Caffe2

---

# Historical Note

These slides are from 2017.

Today:

- PyTorch dominates AI research
- TensorFlow usage has declined significantly in research
- Keras is integrated into TensorFlow
- Caffe is largely legacy
- Caffe2 was eventually merged into PyTorch infrastructure

For MedNAS:
The important takeaway is not Caffe itself, but understanding how framework design evolved from:

Caffe
    ↓
TensorFlow Static Graphs
    ↓
PyTorch Dynamic Graphs
    ↓
Modern NAS / AutoML Systems


