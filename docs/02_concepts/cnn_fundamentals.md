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























