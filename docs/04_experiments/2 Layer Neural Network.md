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





