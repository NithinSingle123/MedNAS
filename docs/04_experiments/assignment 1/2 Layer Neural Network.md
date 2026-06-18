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

# Forward Pass

Now to start this we have already seen how gates work, e.g., max gate, multiplication gate, addition gate, and so on.

But what we want to know is how it applies to the concept of implementation of Neural Networks

One might say that while looking at the topics of forward and backward pass we saw that the gate have already specified which operation to perform.

The question might arise that, while you are implementing 2 layer neural network you don't really have defined operations, so how are we going to approach this.

## You don't find the operation. The architecture defines it.

Now let us see what this means....

let us say we have a two layer network with the following properties:

```python
TwoLayerNet(
    input_size=4,
    hidden_size=10,
    num_classes=3
)
```
This statement itself defines it all.

Well if you remember we looked at the structure of a neural network it goes something like
```
Input (4)
   ↓
Fully Connected Layer
   ↓
ReLU
   ↓
Fully Connected Layer
   ↓
Scores (3)
```
[[Convolutional Neural Networks#Hierarchy of filters]]

And the above sequence starting from input and ending with calculation of scores is one forward pass.

Note: A forward pass does not mean including Training, what happens in training goes like this...
```
Forward Pass
↓
Compute Loss
↓
Backward Pass
↓
Update Weights
```

Note: A two layer neural network means the count of weight matrices, to be simply put you are counting hidden layer and output layer excluding the input layer that is why it is a 2 layer neural network

Question: Another question arises potentially here which is does the forward pass happen 2 times for a 2 layer neural network ?

And the answer is not quite.....
**One Forward pass means:**
```
Take input
↓
Pass through ALL layers
↓
Get output
```

So for a 2 layer network:
```
Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Output
```

So the natural next assumption would be that the code for forward pass should be something like this..
```python
for y in range of inputs(size): 
hidden = X.dot(W1) + b1 
hidden = ReLU(hidden) 
scores = hidden.dot(W2) + b2
```

But here is where you go wrong this is the code but only for the case where in you are going to process one sample at a time. THIS IS A FORWARD PASS FOR EACH SAMPLE INDIVIDUALLY.

BUT NUMPY LETS US PROCESS ALL SAMPLES SIMULTANEOUSLY (This is called vectorization)

```
X (5,4)
↓
W1 (4,10)
↓
hidden (5,10)
↓
ReLU
↓
(5,10)
↓
W2 (10,3)
↓
scores (5,3)
```

my code assumption for it...
```python
 X.shape = (N, D)
    W1.shape = (D, H)
    b1.shape = (H, )
    hidden = X.dot(W1) + b1
    hidden = hidden(ReLU)
    hidden = X
    X.shape = (N, H)
    W2.shape = (H, C)
    b2.shape = (C, )
    hidden = X.dot(W2) + b2
    hidden = hidden(ReLU)
    hidden = scores
    scores.shape = (N, C)
```

But the above code is rough and in a pseudo format and the .shape does not need to be used repeatedly.
```python
hidden = np.dot(X, W1) + b1  
hidden = np.maximum(0, hidden) # ReLU  
scores = np.dot(hidden, W2) + b2
```
this pretty much covers it...

and with this you will get your difference

# Softmax

[[cnn_fundamentals#Softmax Classifier]]

If you look at what is happening in softmax then you will realize the first step in softmax is to first use the exponential function to avoid dealing with negative probabilities.

the next step is to convert them into probabilities.

my code assumption...
```python
for i in scores
	for j in scores
		scores[i][j] = np.exp(scores[i][j])
		
for j in scores
	sum0+=scores[0][j]
	sum1+=scores[1][j]
	sum2+=scores[2][j]

for i in scores
	for j in scores
		if(i=0)
			scores[i][j]/=sum0
		if(i=1)
			scores[i][j]/=sum1
		if(i=2)
			scores[i][j]/=sum2
			
for i in range y
	if(y[i]==scores[j])
		loss=-np.log(scores[i][j])
```

thought process here...

```
Take scores
↓
Apply exp() to every score
↓
Compute row sums
↓
Divide each score by its row sum
↓
Get probabilities
↓
Look up correct class probability
↓
Take -log()
↓
Get loss
```

# Mistakes in My First Softmax Loss Attempt

## 1. Iterating Over `scores` Incorrectly

### Problem

I wrote:
```python
for i in scores:
    for j in scores:
```

Here, `i` and `j` are entire rows of the matrix, not row and column indices.

Since:
```python
scores.shape = (N, C)
```

I need numerical indices to access individual elements.

### Fix

Use index ranges instead:

```python
for i in range(N):
    for j in range(C):
```

## 2. Hardcoding Class Sums

### Problem

I wrote:
```python
sum0 += scores[0][j]
sum1 += scores[1][j]
sum2 += scores[2][j]
```

This only works if there are exactly 3 classes.

A neural network should work for any number of classes:
```python
C = 3
C = 10
C = 100
...
```

### Fix

Compute the sum for each row dynamically:

```python
row_sum = 0
for j in range(C):
    row_sum += scores[i][j]
```

## 3. Summing the Wrong Dimension

### Problem

I was treating:
```python
scores
```

as if there were only three rows.
Softmax works **row-wise**.
Each row represents one training example:
```python
scores[i]
```
and must be converted into a probability distribution independently.

### Fix

For every sample:
```python
for i in range(N):
```

compute:

$$
\sum_j e^{s_j}
$$

using only the scores from that row.

## 4. Misunderstanding the Meaning of `y`

### Problem

I wrote:
```python
if(y[i] == scores[j])
```

But:
```python
y
```
contains class labels, not score values.

Example:
```python
y = [0, 1, 2, 2, 1]
```

means:
```text
Sample 0 belongs to Class 0
Sample 1 belongs to Class 1
Sample 2 belongs to Class 2
...
```

### Fix

Treat:
```python
y[i]
```
as a column index.

Example:
```python
y[0] = 1
```

means:
```text
Column 1 contains the correct class probability.
```

## 5. Not Extracting the Correct-Class Probability

### Problem

I was trying to compare labels and scores directly.
Softmax Loss only needs the probability assigned to the correct class.

### Fix

Use:
```python
scores[i][y[i]]
```

Example:
```python
scores[0] = [0.13, 0.87, 0.01]
y[0] = 1
```

Then:
```python
scores[0][y[0]]
```

becomes:
```python
scores[0][1]
```

which gives:
```python
0.87
```

This is the probability of the correct class.


# What I Got Right

✓ Realized Softmax must convert scores into probabilities.
✓ Realized exponentials (`exp`) are involved.
✓ Realized probabilities must sum to 1.
✓ Understood the need for applying log
✓ Understood the overall pipeline:

```text
Scores
↓
exp()
↓
Normalize
↓
Probabilities
↓
Correct Class Probability
↓
-log(...)
↓
Loss
```

These are the core ideas behind Softmax Loss.

Now given below is the correct code for implementation for the softmax...
```python
exp_scores = np.exp(scores)  
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  
  
correct_logprobs = -np.log(probs[np.arange(N), y])  
  
data_loss = np.sum(correct_logprobs) / N  
reg_loss = reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  
  
loss = data_loss + reg_loss
```

the only thing is that above we have also covered for regularization
Note: reg here is regularization strength
Regularization is nothing but penalty for large weights, just to discourage overfitting

[[cnn_fundamentals#L2 Regularization]]

---

# Backward Pass

# Question: Are We Supposed to Use Jacobians in Backpropagation Since Everything is in Matrix Form?

## Short Answer

**Technically yes, practically no.**

## The Mathematically Correct Answer

Suppose:

$$  
\mathbf{y}=f(\mathbf{x})  
$$

where:

$$  
\mathbf{x}\in\mathbb{R}^n  
$$

and

$$  
\mathbf{y}\in\mathbb{R}^m  
$$

The derivative is no longer a single number.

Instead, it becomes a **Jacobian Matrix**:
$$  
J=  
\frac{\partial \mathbf{y}}  
{\partial \mathbf{x}}  
$$

### Example

Let:
# $$  
\mathbf{x}

\begin{bmatrix}  
x_1\  
x_2  
\end{bmatrix}  
$$
and

# $$  
\mathbf{y}

\begin{bmatrix}  
y_1\  
y_2  
\end{bmatrix}  
$$

Then:
$$  
J=  
\begin{bmatrix}  
\frac{\partial y_1}{\partial x_1}  
&  
\frac{\partial y_1}{\partial x_2}  
\  
\frac{\partial y_2}{\partial x_1}  
&  
\frac{\partial y_2}{\partial x_2}  
\end{bmatrix}  
$$

Therefore:

```text
Scalar → Scalar  => Ordinary Derivative

Vector → Vector  => Jacobian Matrix
```

---

## Why Doesn't CS231n Use Jacobians Explicitly?

Because Jacobians become enormous.

Example:
```python
scores.shape = (5, 3)
```

already creates a sizeable Jacobian.

For realistic neural networks:
```python
X.shape = (128, 3072)
W.shape = (3072, 100)
```

the Jacobians become absurdly large.
Computing and storing them would be extremely expensive.

---

## What Backpropagation Does Instead

Backpropagation uses a trick called:
### Vector-Jacobian Product (VJP)

Instead of explicitly building:
$$  
J  
$$

we directly compute:
$$  
g^T J  
$$

where:
$$  
g=  
\frac{\partial L}  
{\partial y}  
$$

is the upstream gradient.

---

## Why Stanford Says

```text
Upstream Gradient
×
Local Derivative
```

instead of:

```text
Construct Jacobian
Multiply Jacobian
```

The Jacobian math is still happening.

Backprop simply performs the multiplication efficiently without ever constructing the full Jacobian matrix.

---

### Example: Addition Gate

Forward:

$$
z = x + y
$$

The Jacobian is:

$$
\frac{\partial z}{\partial (x,y)}
=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

Suppose the upstream gradient is:

$$
\frac{\partial L}{\partial z}=5
$$

Then:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial z}
\cdot
\frac{\partial z}{\partial x}
=
5 \cdot 1
=
5
$$

and

$$
\frac{\partial L}{\partial y}
=
\frac{\partial L}{\partial z}
\cdot
\frac{\partial z}{\partial y}
=
5 \cdot 1
=
5
$$

Therefore:

$$
\frac{\partial L}{\partial x}=5
$$

$$
\frac{\partial L}{\partial y}=5
$$

Same result as using the Jacobian, but much simpler.

---

## Example: Matrix Multiplication

Forward:

$$  
H=XW  
$$

Technically:

$$  
\frac{\partial H}{\partial W}  
$$

is a massive Jacobian tensor.

Nobody computes it directly.

Instead, backprop derives compact matrix expressions:

$$  
dW=X^T dH  
$$

and

$$  
dX=dH W^T  
$$

These formulas are mathematically equivalent to Jacobian operations but avoid constructing the Jacobian explicitly.

---

## Key Insight

Whenever you see:

```python
dW = X.T.dot(dH)
```

or

```python
dX = dH.dot(W.T)
```

you are secretly doing Jacobian mathematics.

The derivation comes from Jacobians.

The implementation uses matrix calculus and vector-Jacobian products for efficiency.

---

## Mental Model

```text
Single Variable
↓
Ordinary Derivative

Vector → Vector
↓
Jacobian

Deep Learning
↓
Vector-Jacobian Products (VJP)

PyTorch
↓
Automatically performs VJPs
during loss.backward()
```

---

## Final Takeaway

When implementing Assignment 1:

```text
Do I need to explicitly construct Jacobian matrices?
```

**No.**

```text
Am I still using Jacobian mathematics?
```

**Yes.**

The compact matrix formulas used in CS231n are simply an efficient way of performing the same computations that full Jacobians would produce.

# Backward Propagation

So looking at the flowchart above we can definitely tell that while the forward pass what happened is mentioned below:
$$ scores = W_2 * hidden$$
Now using the same expression we can also say,
$$ dW_2 = hidden^T * dscores$$
and similarly, $$dhidden = dscores*W_2^T$$
Above you can see a pattern of sorts emerging....
Here is an important intuition:

```
Forward:

hidden
   │
   ▼
  W2
   │
   ▼
scores
```

```
Backward:

dscores
    │
    ▲
   W2
    │
    ▲
dhidden
```

Here is what is more intrinsically happening here:
```
Hidden Layer (10 neurons)
        ↓
        W2
        ↓
Output Layer (3 scores)
```

The gradient arriving at a hidden neuron is the sum of all the gradient contributions coming from the output neurons it connects.

During Backpropagation:
```
Gradient from Class0
+
Gradient from Class1
+
Gradient from Class2
=
Gradient at Hidden5
```

which is exactly why:
```
dhidden = dscores.dot(W2.T)
```
---

**A Question might arise that there are two gate addition and multiplication, here we are using addition then where are we using multiplication ??**

The answer to this question lies along the principles of the nature of the operations

*MULTIPLY  ALONG THE PATHS  and  ADD ACROSS PATHS*
Lets Compare here for a sec:

One path
```
h
↓
q
↓
L
```

Derivative:$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial q}*\frac{\partial q}{\partial h}$$
Here we multiply as you can see.

Three paths
```
      → c0 →
h →
      → c1 →

      → c2 →
```

Derivative: $$ \frac{\partial L}{\partial h} = Contribution through c0 + Contribution through c1 + Contribution through c2 $$
Here we add.

---

To make things much more practical lets look at it from the perspective of the neural network...

Lets take one path:
```
Hidden5
   │
   ▼
Class0 Score
   │
   ▼
Loss
```

Here, h = Hidden5, s0 = Class0 score, L = Loss

the graph will look will look something like this,
```
h --> s0 --> L
```

Now, $$ \frac{\partial L}{\partial h} = \frac{\partial L}{\partial s_0}*\frac{\partial s_0}{\partial h}$$
also equal to,$$ ds_0 * W2[5, 0]$$

Notice the multiplication...
In simple terms this is nothing but the 
**upstream gradient * local derivative**

Now there are 3 such classes in this case,
```
      ----> s0
     |   
h --------> s1
	 |
	  ----> s2
```

since hidden5 affects all three paths:
$$ \frac{\partial L}{\partial h} =  ds_0 * W2[5, 0] + ds_0 * W2[5, 0] + ds_0 * W2[5, 0] $$

So our rough network just to keep in mind would be this:
```
X
↓
Affine 1
↓
ReLU
↓
hidden
↓
Affine 2
↓
scores
↓
Softmax Loss
↓
L
```

So our affine 2 will be something like:
```
dW2     = hidden.T.dot(dscores)
db2     = np.sum(dscores, axis=0)
dhidden = dscores.dot(W2.T)
```

After this we have to get through ReLU as well, So our expanded backward pass will be:
```
Loss
↑
dscores
↑
Affine 2 Backward
↑
dhidden_after_relu
↑
ReLU Backward
↑
dhidden
↑
Affine 1 Backward
↑
dW1, db1, dX
```

---
But then a question might arise that there are only 2 layers in this network (input is not counted), so hidden and output layers, but why are we going backward 2 times?

Here the answer is that instead of counting neuron groups count weight matrices.
```
Input
↓ W1
Hidden
↓ W2
Output
```
Now, here you can see that there are 2 affine/weight layers, so the backpropagation must compute gradients for both W1,b1 and W2,b2

**Note: When you are dealing with numpy and it mentions something like (3, ) in the case of shape of bias, it doesn't mean 3 rows and 0 columns, INSTEAD, it means A 1-D array with 3 elements. IT HAS NO COLUMN/ROW ORIENTATION.**

*Here numpy says I know there are 3 numbers but I don't know whether you mean a row vector or a column vector *
```python
a.shape = (3,)
[10 20 30]
```

Now let us look at the line
```
db2 = np.sum(dscores, axis=0)
```

Visual situation is something like this:
```
      Class0 Class1 Class2

Row1    a      b      c
Row2    d      e      f
Row3    g      h      i
Row4    j      k      l
Row5    m      n      o
-------------------------
Sum     ?      ?      ?
```

summing along axis 0 means go down each column
The intuition behind this:
```
Bias Gradient Rule

A bias is shared across all samples.

Therefore its gradient is the sum
of the gradients contributed by
every sample.

Hence:

db = np.sum(dout, axis=0)
```

Next up we will see how to pass through ReLU backward

In forward what happens is that, if a neuron has a negative value then it gets killed and output is 0 but now imagine a gradient arrived...
SHOULD THAT GRADIENT BE ALLOWED THROUGH?

HINT:
$$ \frac{d}{dx}\max(0, x) = \begin{cases} 1 & x>0 \\ 0 & x\leq 0 \end{cases}$$
so for every entry hidden_after_ReLU > 0 gradient passes and for every enty hidden_after_ReLU<=0 gradient becomes zero.

Again to keep in reference the actual flow of the graph is as follows:
```
X
↓
XW1+b1
↓
hidden
↓
ReLU
↓
hidden_after_relu
↓
XW2+b2
↓
scores
↓
loss
```

Now the total code implementation of the backpropagation for this neural network is as follows:
```python
# Softmax backward  
dscores = probs  
dscores[np.arange(N), y] -= 1  
dscores /= N

# Layer 2 backward
dW2 = hidden.T.dot(dscores)
db2 = np.sum(dscores, axis=0)
dhidden = dscores.dot(W2.T)

# ReLU backward
dhidden[hidden <= 0] = 0

# Layer 1 backward
dW1 = X.T.dot(dhidden)
db1 = np.sum(dhidden, axis=0)
dX = dhidden.dot(W1.T)

# Regularization gradients
dW2 += reg * W2
dW1 += reg * W1

# Store gradients
grads['W1'] = dW1
grads['b1'] = db1
grads['W2'] = dW2
grads['b2'] = db2
```

## Now let us understand where dscores come from...

# Step 1: What is dscores?

By definition:

$$dscores=\frac{\partial L}{\partial scores}$$

Read it as:

```
How much does each score affect the final loss?
```

---

Suppose after the forward pass we got:

```
scores =[ [2.0, 5.0, 1.0]]
```

One sample.

Three classes.

---

# Step 2: Apply Softmax

Softmax converts scores into probabilities.

Suppose:

```
probs =[ [0.04, 0.94, 0.02]]
```

Meaning:

```
Class 0 : 4%Class 1 : 94%Class 2 : 2%
```

---

Suppose the true label is:

```
y = [1]
```

Meaning:

```
Class 1 is correct.
```

---

# Step 3: What Does Loss Want?

The loss wants:

```
Correct class probability↑Wrong class probabilities↓
```

For this example:

```
Current:[0.04, 0.94, 0.02]
```

Loss says:

```
Good.Make Class 1 even higher.Make others lower.
```

---

# Another Example

Suppose:

```
probs =[ [0.30, 0.40, 0.30]]
```

and:

```
y = [1]
```

Again Class 1 is correct.

Loss wants:

```
Increase 0.40Decrease 0.30Decrease 0.30
```

---

# The Amazing Result

If you derive:

```
Softmax+Cross Entropy
```

with calculus,

everything collapses into:

$$\frac{\partial L}{\partial scores}=probs - one_{hot}(y)$$

This is one of the most beautiful results in deep learning.

---

# What is one_hot(y)?

If:

```
y = 1
```

then:

```
one_hot(y)=[0,1,0]
```

because class 1 is correct.

---

Therefore:

```
probs =[0.30, 0.40, 0.30]one_hot =[0.00, 1.00, 0.00]
```

Subtract:

```
dscores =[0.30, -0.60, 0.30]
```

---

Interpretation:

```
Class 0:positive gradient↓push score downClass 1:negative gradient↓push score upClass 2:positive gradient↓push score down
```

Exactly what we wanted.

---

# Now The Famous Code

First:

```
dscores = probs
```

Now:

```
dscores =[ [0.30, 0.40, 0.30]]
```

---

Then:

```
dscores[np.arange(N), y] -= 1
```

Suppose:

```
N = 1y = [1]
```

Then:

```
dscores[0,1] -= 1
```

which gives:

```
[ [0.30, -0.60, 0.30]]
```

Exactly:

```
probs - one_hot(y)
```

without explicitly constructing the one-hot vector.

---

Then:

```
dscores /= N
```

because the loss was averaged over the batch:

$$
Loss=\frac{1}{N}\sum_{i=1}^{N}L_i$$

Therefore gradients must also be averaged.

---

# The Big Picture

Forward:

```
scores↓softmax↓probs↓cross entropy↓loss
```

Backward:

```
loss↑cross entropy↑softmax↑dscores
```

And the miracle is:

```
dscores = probsdscores[np.arange(N), y] -= 1dscores /= N
```

already computes:

$$\frac{\partial L}{\partial scores}$$​

without you ever needing to explicitly differentiate the giant Softmax formula.

After pages of calculus, Softmax + Cross Entropy backward collapses to those three lines. That's why they're famous.


# Training Mini batches

So let us now look at why do we need mini batches ??
Suppose X.shape = (49000, 3072)
that means all 49000 images and if every gradient update uses all 49000 images (compute loss, compute gradients, update weights) it becomes very slow...

Instead here is the flow of what we do...
```
Pick 200 random images
↓
Compute loss
↓
Compute gradients
↓
Update weights
```
This is called the Stochastic Gradient Descent (SGD)

Now let us look at how we can do that..
```python
np.random.choice(
    num_train,
    batch_size
)
```
Here num_train is 49000 and batch_size is 200, and conceptually this means choosing 200 indices from the 49000 training examples.

Now what the above implementation gives us is an array of indices of shape (200, ). Now which numpy operation are we going to use to extract the 200 selected rows from X

Now the first thing that comes to mind is to use an iterative loop, but numpy has a fancier way of addressing this...

Example:
```python
import numpy as np
X = np.array(["row0", "row1", "row2", "row3", "row4", "row5"])
indices = [1,3]
X[indices]
```
output:
row1, row3

So in this case X_batch = X[indices] gives shape = (200, 3072) automatically.
And similarly y_batch = y[indices] gives shape = (200, )

so the code implementation for the entire minibatch selection is basically:
```python
indices = np.random.choice(
    num_train,
    batch_size,
    replace=True
)

X_batch = X[indices]
y_batch = y[indices]
```

## Question
DO YOU THINK WE SHOULD ALLOW THE SAME TRAINING EXAMPLE TO BE PICKED TWICE IN ONE MINIBATCH??

Intuitively the answer would be no saying it should be unique obviously...

**For SGD, duplicates are actually allowed and commonly used**

This is because we are not trying to create a perfect sample instead we are trying to create a random estimate of the full gradient.
```python
replace = True
```

# Why Do We Use Mini-Batches Instead of the Full Dataset?

The goal of SGD is not:

```text
Compute the exact gradient.
```

The goal is:

```text
Get to a good set of weights
as fast as possible.
```

---

## Full Batch Gradient Descent

Suppose we are training on CIFAR-10:

```python
49000 training images
```

If we compute the gradient using the entire dataset:

```text
49000 images
↓
Compute exact gradient
↓
One weight update
```

Cost:

```text
49000 image evaluations
```

for a single update.

---

## Mini-Batch Gradient Descent (SGD)

Instead:

```text
200 images
↓
Approximate gradient
↓
One weight update
```

Cost:

```text
200 image evaluations
```

for a single update.

---

## Comparison

In the time Full Batch Gradient Descent performs:

```text
1 update
```

Mini-Batch SGD can perform roughly:

$$
\frac{49000}{200}
=
245
$$

updates.

Therefore:

```text
Exact gradient
↓
1 update

vs

Approximate gradient
↓
245 updates
```

In practice, SGD usually reaches a good solution much faster.

---

## But Isn't The Exact Gradient Better?

Yes.

The exact gradient is more accurate.

Suppose the true gradient points:

```text
←
```

A mini-batch gradient may point:

```text
↖
```

or:

```text
↙
```

because of sampling noise.

However, it is still approximately pointing:

```text
left
```

which is usually good enough to make progress.

---

## Mountain Analogy

Imagine you are standing on a mountain and want to reach the bottom.

### Strategy 1: Perfect Direction

```text
Spend 2 hours calculating
the exact best direction

↓

Take 1 step
```

### Strategy 2: Fast Estimates

```text
Take a quick estimate

↓

Take a step

↓

Repeat 200 times
```

The second person will usually reach the bottom first.

This is exactly why SGD is preferred.

---

## Why Are Duplicates Allowed?

Suppose:

```python
batch_size = 200
```

and a sampled batch contains:

```text
Image #17
Image #17
Image #305
Image #1024
...
```

The gradient estimate becomes slightly noisier.

However:

```text
Dataset Size = 49000
Batch Size   = 200
```

means we are already using an approximation.

The small effect of duplicates is negligible.

Therefore CS231n commonly uses:

```python
indices = np.random.choice(
    num_train,
    batch_size,
    replace=True
)
```

which allows duplicate samples.

---

## Deep Learning Secret

Modern deep learning models are trained using:

```text
Mini-Batches
```

not:

```text
Full-Batch Gradient Descent
```

because:

```text
Full Dataset
↓
Exact Gradient
↓
Slow

Mini-Batch
↓
Approximate Gradient
↓
Much Faster
```

The approximation is usually accurate enough that training still converges successfully.

---

## Final Takeaway

```text
Goal:
Reach good weights quickly

Not:
Compute the perfect gradient

Tradeoff:

A little gradient noise
↓
Massive speed improvement

Result:
Mini-Batch SGD wins
```


# TODO — SGD Parameter Update

## Objective

Use the gradients computed during backpropagation to update the network parameters using Stochastic Gradient Descent (SGD).

### Theory

SGD updates parameters according to:

# $$  
W_{new}

W - \eta \frac{\partial L}{\partial W}  
$$

where:

- (W) = parameter
    
- (\eta) = learning rate
    
- (\frac{\partial L}{\partial W}) = gradient


The gradient points in the direction of increasing loss, so we subtract it to move toward lower loss.

### Implementation

```python
self.params['W1'] -= learning_rate * grads['W1']
self.params['b1'] -= learning_rate * grads['b1']
self.params['W2'] -= learning_rate * grads['W2']
self.params['b2'] -= learning_rate * grads['b2']
```

### Key Insight

```text
Current Parameters
        ↓
Compute Gradients
        ↓
Move Opposite Gradient Direction
        ↓
Updated Parameters
```

## This completes the learning step of the neural network.

# TODO — Predict Function

## Objective

Given an input X, predict the class label for each sample.

### Theory

Prediction does not require:

- Loss computation
    
- Backpropagation
    
- Gradients


It only requires a forward pass followed by selecting the class with the highest score.

### Architecture

```text
Input
↓
Affine Layer (W1,b1)
↓
ReLU
↓
Affine Layer (W2,b2)
↓
Scores
↓
argmax
↓
Predicted Class
```

### Implementation

```python
hidden = np.dot(X, self.params['W1']) + self.params['b1']
hidden = np.maximum(0, hidden)

scores = np.dot(hidden, self.params['W2']) + self.params['b2']

y_pred = np.argmax(scores, axis=1)
```

### Why argmax?

Example:

```python
scores = [2.1, 5.4, 1.7]
```

Largest score:

```text
5.4
```

Index:

```text
1
```

Prediction:

```python
y_pred = 1
```

### Key Insight

```text
Training:
Forward → Loss → Backprop → Update

Prediction:
Forward → argmax
```

The predict function is intentionally simple because all learning has already happened during training.
