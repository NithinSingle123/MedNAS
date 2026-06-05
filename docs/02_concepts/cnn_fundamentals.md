# **CNN FUNDAMENTALS**

### **Steps to arrive at a 3d representation of the world**

First we have an input image then we arrive at a step which is called the primal sketch where it has to do with all edges, bars, ends, virtual lines, curves, boundaries.

Then we come to a step called 2 and a half sketch where in we look at local surface orientation, its depth and its associated discontinuities and then FINALLY comes the 3D model the object is finally organized and represented in its 3D form.

# **Image Classification**

### **An Image Classifier**

```python
def classify_image(image):
	some magic here ?
	return class_label
```


Unlike e.g. sorting a list of numbers
no obvious way to hard code the algorithm for recognizing a cat or the other classes

attempts for this have been made like find edges first and then find corners and then use explicit rules to recognition based on rules and edges

instead of sitting down and making rules we go to something called *DATA DRIVEN APPROACH*

1. **collect a dataset of images and labels**
2. **Use ML to train a classifier**
3. **Evaluate the classifier on new images**


```python
def train(images, labels):
	# Machine Learning!
	return model


def predict(model, test_images):
	# Use model to predit labels
	return test_labels
```

then we will use these trained models to recognize new unknown images.

**First Classifer: NEAREST NEIGHBOR**

```python
class NearestNeighbor:

    def __init__(self):
        pass

    def train(self, X, y):
        """
        X is N x D where each row is a training example.
        y is a 1-dimensional array of size N containing labels.
        """

        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        X is N x D where each row is a test example
        for which we want to predict a label.
        """

        num_test = X.shape[0]

        # Make sure output type matches training label type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # Loop over all test examples
        for i in range(num_test):

            # Find the nearest training image to the i-th test image
            # using L1 distance (sum of absolute differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)

            # Get index of smallest distance
            min_index = np.argmin(distances)

            # Predict the label of nearest example
            Ypred[i] = self.ytr[min_index]

        return Ypred

```

we just learn the data and labels and then find the most similar image in the test cases and recognize it

**we use distance metrics like Manhattan or minkowski or euclidean to just compare two different images (their matrices) and the if the difference is least then it would label it under the same category irrespective of whether it is correct visually or not**

very slow at test time and the dist metrics on pixels are not informative
CURSE OF DIMENSIONALITY
## Hyperparameters

what is the best value of k to use ?
what is the best distance to use ?

These are hyperparameters: choices about the algorithm that we set rather than learn

To navigate this the easy way is that we kind of divide the data into three parts *train*, *val*, and *test*; where in we train the data using different distances and then validate it and see which one gives us the optimal results and then move on to applying it to test set


## Linear Classification

like basic lego building blocks for the neural networks

the linear classifier is one of the simplest examples of parametric approach

<mark style="background: #FFF3A3A6;">*IMAGE*  ------> **f(x,W)** ------->*10 numbers giving class scores</mark>
                 A  
                  |
                  |
                  |
#                      W 
        parameters of weights     

#  here,... f(x, W) = Wx + b

Larger score for cat indicates that the image is more probable to be a cat.

<mark style="background: #FF5582A6;">here the array of the image is 32 x 32 x 3 for rgb what if it is black and white then will it be 32 x 32 x 2 ??</mark>

the situation where a linear classifier struggles are the multimodal situations


# LOSS FUNCTIONS AND OPTIMIZATION

example W table below for reference

| template | cat         | car          | frog          |
| -------- | ----------- | ------------ | ------------- |
| plane    | -3.45       | -0.51        | 3.42          |
| car      | -8.87       | ==**6.04**== | 4.64          |
| cat      | ==**2.9**== | -4.22        | 5.1           |
| dog      | 8.02        | 3.58         | 5.55          |
| frog     | 3.78        | 4.49         | ==**-4.34**== |
A loss function is a measure that quantifies our unhappiness with the scores across the training data

here we want the true class score to be the highest and as you can see if we take the case of the cat the correspondent class value is 2.9 which is not the highest means the classifier did bad in the case of cat whereas in the case of car the condition holds.

A loss function tells how good our current classifier is 
Given a dataset of examples,
# $\{(x_i,y_i)\}_{i=1}^{N}$

where $x_i$ is image and
    $y_i$ is (integer) label

Loss over the dataset is a sum of loss over examples:
# $$ L = \frac{1}{N} \sum_i L_i(f(x_i, W), y_i) $$
## Multiclass SVM loss function

Given an example $(x_i, y_i)$  
  
where:  
- $x_i$ is the image  
- $y_i$ is the (integer) label  
  
and using the shorthand for the scores vector:  
  
$$  
s = f(x_i, W)  
$$  
  
the SVM loss has the form:  
  
$$  
L_i = \sum_{j \ne y_i}  
\begin{cases}  
0 & \text{if } s_{y_i} \ge s_j + 1 \\  
s_j - s_{y_i} + 1 & \text{otherwise}  
\end{cases}  
$$  
  
which can also be written as:  
  
$$  
L_i = \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)  
$$


The Ss are the predicted scores for the classes that are coming out of the classifier. For example if 1 is the cat class and 2 is the dog class then S1 and S2 would be the cat and dog scores respectively.

$$S_{yi}$$
the above character corresponds to the score of the true class for the i th example in the training set

a Mathematical Example:


| template | cat     | car   | frog     |
| -------- | ------- | ----- | -------- |
| cat      | 3.2     | 1.3   | 2.2      |
| car      | 5.1     | 4.9   | 2.5      |
| frog     | -1.7    | 2.0   | -3.1     |
| Losses   | ==2.9== | ==0== | ==12.9== |

Given an example $(x_i, y_i)$  
  
where:  
- $x_i$ is the image  
- $y_i$ is the (integer) label  
  
and using the shorthand for the scores vector:  
  
$$  
s = f(x_i, W)  
$$  
  
the SVM loss has the form:  
  
$$  
L_i = \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)  
$$  
  
## Example Calculation  
  
$$  = \max(0, 5.1 - 3.2 + 1)  $$
  
$$  + \max(0, -1.7 - 3.2 + 1)  $$  
  
$$  
= \max(0, 2.9) + \max(0, -3.9)  
$$
$$  
= 2.9 + 0  
$$
$$  
= 2.9  
$$

The final loss for the dataset is the average of the losses obtained for the three L = (2.9 + 0 + 12.9) / 3 = 5.27

The +1 is an arbitrary choice and it doesnt matter and kind of washes out compared to the scale

<mark style="background: #FFB86CA6;">If we jiggle the values a little bit for the car class the what would happen is that you would not find any change, this is because it is already greater than the others the margin of one still retains and we still get zero loss</mark>

<mark style="background: #FFF3A3A6;">the minimum loss you can incur is 0 and max loss you can incur is infinity</mark>

# Initial SVM Loss at Weight Initialization

## Question

At initialization, the weight matrix $W$ is very small, so all scores:

$$
s \approx 0
$$

What is the multiclass SVM loss?

---

## Multiclass SVM Loss Formula

$$
L_i = \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)
$$

---

## Substituting Initial Scores

Since:

$$
s_j \approx 0
$$

and

$$
s_{y_i} \approx 0
$$

each term becomes:

$$
\max(0, 0 - 0 + 1)
=
\max(0, 1)
=
1
$$

---

## Number of Incorrect Classes

For 3 classes:
- 1 correct class
- 2 incorrect classes

Therefore:

$$
L_i = 1 + 1 = 2
$$

---

# Final Answer

$$
L_i = 2
$$

---

# Use as a Debugging Strategy

This acts as a sanity-check debugging technique.

When weights are initialized near zero:
- all class scores become approximately equal
- the initial loss becomes predictable

Therefore:
- expected initial loss $\approx 2$

---

# Why This Is Useful

If the implementation is correct:

$$
L_i \approx 2
$$

should appear at the beginning of training.

If the loss is:
- extremely large
- negative
- zero
- NaN

then there is likely:
- a bug in the loss implementation
- incorrect indexing
- wrong score computation
- numerical instability

---

# Key Insight

In deep learning debugging:

> Understanding expected behavior is extremely important.

Researchers often debug models by checking whether:
- initial losses,
- accuracies,
- gradients,
- score distributions

match theoretical expectations.


```python
def L_i_vectorized(x, y, w):
	scores = W.dot(x)
	margins = np.maximum(0, scores - scores[y] + 1)
	margins[y] = 0
	loss_i = np.sum(margins)
	return loss_i
```


# Regularized Loss Function

$$
L(W) =
\frac{1}{N}
\sum_{i=1}^{N}
L_i(f(x_i, W), y_i)
+
\lambda R(W)
$$

---

# Formula Breakdown

| Term | Meaning |
|---|---|
| $L(W)$ | Total loss |
| $N$ | Number of training examples |
| $L_i$ | Loss for a single example |
| $f(x_i, W)$ | Model prediction |
| $y_i$ | True label |
| $R(W)$ | Regularization term |
| $\lambda$ | Regularization strength |

---

# Two Main Components

## 1. Data Loss

$$
\frac{1}{N}
\sum_{i=1}^{N}
L_i(f(x_i, W), y_i)
$$

Purpose:
- measures how well predictions match training data
- encourages accurate classification

Goal:
- reduce prediction error

---

## 2. Regularization

$$
\lambda R(W)
$$

Purpose:
- penalizes overly complex models
- encourages simpler weight configurations
- improves generalization on unseen data

Goal:
- prevent overfitting

---

# What is Regularization?

Regularization is a technique used to prevent a neural network from:
- memorizing training data
- becoming too complex
- performing poorly on test data

It adds a penalty to the loss function based on the model weights.

---

# Why Regularization Matters

Without regularization:
- the model may fit training data perfectly
- but fail on real-world unseen data

This problem is called:

$$
\text{Overfitting}
$$

---

# Key Intuition

A model should not only:
- fit the training data

but also:
- remain simple enough to generalize well

---

# Common Idea

Large weights often indicate:
- overly sensitive models
- memorization behavior
- unstable decision boundaries

Regularization discourages excessively large weights.

---

# Most Common Regularization

## L2 Regularization

$$
R(W) = \sum W^2
$$

This penalizes large weight values.

---

# Effect of Regularization

Regularization typically:
- reduces overfitting
- improves generalization
- creates smoother models
- stabilizes learning

---

# Tradeoff

| Too Little Regularization | Too Much Regularization |
|---|---|
| Overfitting | Underfitting |
| Memorization | Oversimplification |
| Poor test accuracy | Poor training accuracy |

---

# Important Insight

The total loss function balances:

$$
\text{Prediction Accuracy}
+
\text{Model Simplicity}
$$

This balance is fundamental in machine learning.

---

## Optimization

the first idea on the raw though process would be use all the solutions and substitute them in your loss function. But SPOILER ALERT this algo the so called "RANDOM SEARCH" is shit and should not be used.

```python
# assume X_train is the data where each column is an example (e.g. 3073 x 50000)
# assume Y_train are the labels (e.g. ID array of 50000)
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in xrange(1000):
	w = np.random.randn(10, 3073) * 0.0001 # generate random parameters
	loss = L(X_train, Y_train, W) # get the loss over the entire training set
	if loss < bestloss: # keep track of the best solution
		bestloss = loss
		bestW = W
	print 'in attempt %d the loss was %f, best %f' %(num, lossm bestloss)
	
	
# prints:
# in attempt 0 the loss was 9.40, best 9.40
# in attempt 1 the loss was 8.90, best 8.90
# in attempt 2 the loss was 9.04, best 8.90
# continues for 1000 lines
```


Better than the above is to use the local geometry. you can feel where you are going with little steps you take.

## Follow the slope

In 1-dimension the derivative of a function:

$$
\frac{df(x)}{dx}
=
\lim_{h \to 0}
\frac{f(x+h)-f(x)}{h}
$$
<mark style="background: #FFF3A3A6;">
In multiple dimensions the gradient is the vector of (partial derivatives) along each dimension</mark>

The slope in any direction is the dot product of the direction with the gradient. <mark style="background: #FFB8EBA6;">The direction of steepest descent is the negative gradient.</mark>

# Numerical Gradient Approximation

## Current Weights

$$
W =
\begin{bmatrix}
0.34 \\
-1.11 \\
0.78 \\
0.12 \\
0.55 \\
2.81 \\
-3.10 \\
-1.50 \\
0.33
\end{bmatrix}
$$

Current loss:

$$
L(W) = 1.25347
$$

---

## Perturb One Dimension

Add a small value $h$ to one parameter:

$$
W + h =
\begin{bmatrix}
0.34 \\
-1.11 \\
0.78 + 0.0001 \\
0.12 \\
0.55 \\
2.81 \\
-3.10 \\
-1.50 \\
0.33
\end{bmatrix}
$$

New loss:

$$
L(W+h) = 1.25347
$$

---

# Numerical Gradient Formula

$$
\frac{d f(x)}{dx}
=
\lim_{h \to 0}
\frac{f(x+h)-f(x)}{h}
$$

---

# Gradient Vector

$$
dW =
\begin{bmatrix}
-2.5 \\
0.6 \\
? \\
? \\
? \\
? \\
? \\
?
\end{bmatrix}
$$

---

# Key Intuition

A gradient measures:

> How much the loss changes when a parameter changes slightly.

Each weight dimension is perturbed independently to estimate:
- sensitivity
- direction of improvement
- optimization behavior

---

# Why This Matters

Numerical gradients are commonly used to:
- debug backpropagation
- verify gradient correctness
- detect implementation bugs

This process is called Gradient checking

This process might actually be super slow if its a large CNN and this parameter W will not 10 entries like here, it might have millions. so practically you never wanna calculate gradients

# Analytic Gradient Computation

The loss is a function of the weights:

$$
L =
\frac{1}{N}
\sum_{i=1}^{N} L_i
+
\sum_k W_k^2
$$

---

# Multiclass SVM Loss

$$
L_i =
\sum_{j \ne y_i}
\max(0, s_j - s_{y_i} + 1)
$$

---

# Score Function

$$
s = f(x;W) = Wx
$$

---

# Goal

We want to compute:

$$
\nabla_W L
$$

which represents:

> the gradient of the loss with respect to the weights.

---

# Key Idea

Instead of:
- changing each weight individually,
- recomputing the loss repeatedly,

we use:
# calculus

to compute the gradient directly.

This is called:

$$
\text{Analytic Gradient}
$$

---

# Why Analytic Gradients Matter

Numerical gradients are:
- slow
- computationally expensive
- impractical for large neural networks

Analytic gradients are:
- fast
- exact
- scalable

---

# Core Insight

Neural network training depends on:

$$
\nabla_W L
$$

because gradients tell us:
- which direction reduces loss
- how strongly each parameter affects predictions

---

# Relation to Backpropagation

Backpropagation is essentially:
- efficient analytic gradient computation
- using the chain rule repeatedly through layers

---

# Why This Matters for Deep Learning

Without analytic gradients:
- training deep networks would be extremely slow
- optimization would become impractical

Backpropagation makes modern deep learning possible.

---

# Important Transition

The workflow evolves from:

## Numerical Gradient

$$
\frac{f(x+h)-f(x)}{h}
$$

to:

## Analytic Gradient

$$
\nabla_W L
$$

which is:
- mathematically derived
- much faster
- used during real training


## Gradient Descent

So gradient descent is first we initialize our W as some random thing while true we will update our loss and gradient and we will update our weights in the opposite of the gradient direction, and doing this will converge your network and give you the desired optimal solution

```python
# Vanilla Gradient Descent

while True:
	weights_grad = evaluate_gradient(loss_fun, data, weights)
	weights += - step_size * wights_grad # perform parameter update
```

step size also called the learning rate is one of the single most important parameter that you have to set in practice.


# Stochastic Gradient Descent (SGD)

## Total Loss Function

$$
L(W) =
\frac{1}{N}
\sum_{i=1}^{N}
L_i(x_i, y_i, W)
+
\lambda R(W)
$$

---

# Gradient of the Loss

$$
\nabla_W L(W) =
\frac{1}{N}
\sum_{i=1}^{N}
\nabla_W L_i(x_i, y_i, W)
+
\lambda \nabla_W R(W)
$$

---

# Problem with Full Gradient Computation

Computing gradients over:
- the entire dataset
- every iteration

becomes extremely expensive when:

$$
N
$$

is very large.

---

# Solution — Minibatch Approximation

Instead of using:
- all training examples,

SGD uses:
# a minibatch

of examples to approximate the gradient.

Typical minibatch sizes:

$$
32,\ 64,\ 128
$$

---

# Vanilla Minibatch Gradient Descent

```python
while True:

    # sample minibatch
    data_batch = sample_training_data(data, 256)

    # compute gradient
    weights_grad = evaluate_gradient(
        loss_function,
        data_batch,
        weights
    )

    # update weights
    weights += -step_size * weights_grad
```

---

# Key Intuition

SGD repeatedly:
1. samples a small batch
2. computes approximate gradients
3. updates weights
4. moves toward lower loss

---

# Why SGD Works

Even though minibatch gradients are noisy:
- they are much faster
- require less memory
- still approximate the true optimization direction

This makes deep learning scalable.

---

# Important Components

| Component | Meaning |
|---|---|
| Minibatch | Small subset of training data |
| Gradient | Direction of steepest increase |
| Step Size | Learning rate |
| Weight Update | Parameter optimization step |

---

# Weight Update Rule

$$
W \leftarrow W - \eta \nabla_W L
$$

where:

| Symbol | Meaning |
|---|---|
| $W$ | weights |
| $\eta$ | learning rate |
| $\nabla_W L$ | gradient |

---

# Why This Matters

Modern deep learning relies heavily on:
# stochastic optimization

because full-dataset optimization is often:
- too slow
- too memory-intensive
- computationally impractical

---

# Connection to MedNAS

Future NAS systems will repeatedly:
- train architectures
- compute gradients
- optimize weights
- evaluate performance

Understanding SGD is foundational for:
- architecture search
- optimization dynamics
- training efficiency
- deployment-aware learning


# Image Features — Motivation

## Problem

Some datasets cannot be separated using a simple linear classifier.

Example:
- red points and blue points overlap in the original coordinate space
- a straight line cannot separate them

This means the data is:
# not linearly separable

---

# Feature Transformation

Apply a transformation:

$$
f(x,y) = (r(x,y), \theta(x,y))
$$

This converts:
- Cartesian coordinates
- into polar-coordinate-style features

---

# Key Idea

Instead of changing:
- the classifier,

we change:
# the representation of the data.

---

# Before Feature Transformation

In original space:

$$
(x,y)
$$

the classes overlap.

A linear classifier struggles because:
- boundaries are complex
- relationships are nonlinear

---

# After Feature Transformation

In transformed feature space:

$$
(r,\theta)
$$

the data becomes easier to separate.

Now:
- a simple linear classifier works

---

# Important Insight

Good features can make:
- difficult problems simple

Bad features can make:
- simple problems difficult

---

# Core Machine Learning Principle

A classifier is only as good as:
# the representation of the data.

Feature engineering transforms data into forms where:
- patterns become easier to learn
- separation becomes simpler
- optimization improves

---

# Relation to Neural Networks

Deep learning automates:
# feature extraction

Instead of manually designing transformations,
CNNs learn hierarchical features automatically.

Examples:
- edges
- textures
- shapes
- object parts
- semantic patterns

---

# Why This Matters

The power of deep learning comes largely from:
# representation learning

not just classification itself.

---

# Connection to CNNs

Convolution layers learn feature maps that progressively transform:
- raw pixels
- into meaningful representations

This enables:
- linear separability
- improved classification
- hierarchical understanding

---

# Connection to MedNAS

Future NAS systems may optimize:
- feature extraction strategies
- representation quality
- architectural transformations

because architecture design strongly affects:
# learned feature representations.
---

# Softmax Classifier
## Understanding Softmax Intuitively

Let's start with the problem Softmax is solving.

Suppose your network outputs:

```python
scores = [3.2, 5.1, -1.7]
```

What do these mean?

Honestly: **nothing yet**.

All we know is:

- Class 0 got score 3.2
- Class 1 got score 5.1
- Class 2 got score -1.7

But can we say:

> There is an 80% chance it's Class 1

No.

Because:

- 3.2 is not a probability
- 5.1 is not a probability
- -1.7 is definitely not a probability

Probabilities must satisfy:

$$
0 \le P \le 1
$$

and

$$
\sum_i P_i = 1
$$

Our scores satisfy neither.

Therefore Softmax asks:

> Can I turn these arbitrary scores into probabilities?

For example:

```python
scores = [3.2, 5.1, -1.7]
```

should become something like:

```python
probs = [0.13, 0.86, 0.01]
```

Now we can say:

- 13% Cat
- 86% Car
- 1% Frog

which actually makes sense.

---

### Naive Attempt

Suppose we simply divide by the total:

```python
scores = [3.2, 5.1, -1.7]
sum = 6.6
```

giving:

```python
[0.48, 0.77, -0.25]
```

Problem:

> Negative probability!

Impossible.

---

### Softmax Solution

Softmax first applies the exponential function:

$$
e^x
$$

Why?

Because exponentials are always positive.

For our scores:

```python
scores = [3.2, 5.1, -1.7]
```

we get:

$$
[e^{3.2}, e^{5.1}, e^{-1.7}]
$$

approximately:

```python
[24.5, 164.0, 0.18]
```

Notice:

> Everything is positive now.

---

### Normalize into Probabilities

Add everything:

$$
24.5 + 164.0 + 0.18 = 188.68
$$

Now divide each value by the total:

$$
\frac{24.5}{188.68}=0.13
$$

$$
\frac{164.0}{188.68}=0.87
$$

$$
\frac{0.18}{188.68}=0.001
$$

Result:

```python
[0.13, 0.87, 0.001]
```

Properties:

- No negative values
- Values lie between 0 and 1
- Sum equals 1

Therefore:

```python
[0.13, 0.87, 0.001]
```

is a valid probability distribution.

---

### Softmax Formula

$$
\text{Softmax}(s_i)
=
\frac{e^{s_i}}
{\sum_j e^{s_j}}
$$

Interpretation:

1. Convert every score into a positive number using $e^x$
2. Divide by the total
3. Obtain probabilities

Result:

> Arbitrary scores → Probability distribution

---

### Connecting Softmax to Labels

Suppose:

```python
scores = [3.2, 5.1, -1.7]
```

becomes:

```python
probs = [0.13, 0.87, 0.001]
```

and the true label is:

```python
y = 1
```

which means:

> Class 1 is the correct class.

Looking at:

```python
probs = [0.13, 0.87, 0.001]
```

we have:

| Class | Probability |
|---------|---------|
| 0 | 0.13 |
| 1 | 0.87 ← Correct Class |
| 2 | 0.001 |

The loss function only cares about:

$$
P(\text{correct class})
=
0.87
$$

because `y = 1`.

This idea leads directly to Softmax Loss:

$$
L_i
=
-\log\left(P(\text{correct class})\right)
$$




