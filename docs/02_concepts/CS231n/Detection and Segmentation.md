![[Screenshot 2026-06-11 025455 1.png]]


## Semantic Segmentation

![[Pasted image 20260611025816.png]]

![[Pasted image 20260612010540.png]]

- Problem:
	- Very inefficient! Not reusing shared features between overlapping patches.

![[Pasted image 20260612010622.png]]

**What is the loss function here** 
So here since we are making the classification decision per pixel then we put a cross entropy loss on every pixel of the output. So we have the ground truth category label for every pixel in the output then we compute our cross entropy loss between every pixel in the output and the ground truth pixels and then take either a sum or an average over space and then sum or average over the mini batch.

- Problem:
	- Here the problem is that it is computationally very expensive to apply convolutional layers especially in this high resolution preserving spatial dimensions and filter numbers such 64, 128, 256 (normally seen in these kinds of networks), it would take a ton of memory.

#### Instead what you will see you want to perform convolution in the optimal way is in the image below....

![[Pasted image 20260612010730.png]]

## Unpooling

![[Pasted image 20260612010822.png]]

## Max Pooling

![[Pasted image 20260612010943.png]]

## Transpose Convolution

![[Pasted image 20260612011016.png]]

![[Pasted image 20260612011057.png]]

![[Pasted image 20260612011139.png]]

![[Pasted image 20260612011230.png]]

![[Pasted image 20260612011307.png]]

![[Pasted image 20260612011339.png]]

![[Pasted image 20260612011413.png]]

![[Pasted image 20260612011518.png]]

![[Pasted image 20260612011559.png]]

![[Screenshot 2026-06-12 011700.png]]

**Where does the name transpose convolution come from?**
It comes from the representation of convolution as given below...

# Understanding Transpose Convolution Through Linear Algebra

## Why Was Transpose Convolution Invented?

In semantic segmentation, CNNs repeatedly downsample feature maps:

```text
256×256
 ↓
128×128
 ↓
64×64
 ↓
32×32
```

Eventually we need to recover:

```text
32×32
 ↓
64×64
 ↓
128×128
 ↓
256×256
```

A normal convolution naturally compresses information.

Transpose convolution performs the opposite operation: expanding information.

---

# Step 1: Represent Convolution as Matrix Multiplication

Suppose our input is:

$$
x =
\begin{bmatrix}
1 \\
2 \\
3 \\
4
\end{bmatrix}
$$

We want convolution to produce:

$$
y =
\begin{bmatrix}
5 \\
25
\end{bmatrix}
$$

Define:

$$
K =
\begin{bmatrix}
1 & 2 & 0 & 0 \\
0 & 0 & 3 & 4
\end{bmatrix}
$$

Then:

$$
y = Kx
$$

Substituting:

$$
\begin{bmatrix}
5 \\
25
\end{bmatrix}
=
\begin{bmatrix}
1 & 2 & 0 & 0 \\
0 & 0 & 3 & 4
\end{bmatrix}
\begin{bmatrix}
1 \\
2 \\
3 \\
4
\end{bmatrix}
$$

Computing each row:

$$
y_1 = 1(1)+2(2)+0(3)+0(4)=5
$$

$$
y_2 = 0(1)+0(2)+3(3)+4(4)=25
$$

Result:

$$
y =
\begin{bmatrix}
5 \\
25
\end{bmatrix}
$$



# Step 2: Observe the Dimensions

Input:

$$
x \in \mathbb{R}^{4}
$$

Output:

$$
y \in \mathbb{R}^{2}
$$

Therefore:

$$
K \in \mathbb{R}^{2\times4}
$$

Dimension flow:

```text
4 values
   ↓
   K
   ↓
2 values
```

Convolution compresses information.



# Step 3: The Reverse Problem

Suppose we now have:

$$
y =
\begin{bmatrix}
5 \\
25
\end{bmatrix}
$$

and want to expand back to a vector of length 4.

We need a matrix that maps:

```text
2 values
   ↓
 ?
   ↓
4 values
```

The transpose of K provides exactly this.



# Step 4: Compute the Transpose

Original matrix:

$$
K =
\begin{bmatrix}
1 & 2 & 0 & 0 \\
0 & 0 & 3 & 4
\end{bmatrix}
$$

Transpose:

$$
K^T =
\begin{bmatrix}
1 & 0 \\
2 & 0 \\
0 & 3 \\
0 & 4
\end{bmatrix}
$$

Rows become columns.

Dimensions become:

$$
K^T \in \mathbb{R}^{4\times2}
$$



# Step 5: Apply the Transpose

Compute:

$$
x' = K^T y
$$

Substitute:

$$
x' =
\begin{bmatrix}
1 & 0 \\
2 & 0 \\
0 & 3 \\
0 & 4
\end{bmatrix}
\begin{bmatrix}
5 \\
25
\end{bmatrix}
$$

Calculating:

$$
x'_1 = 1(5)+0(25)=5
$$

$$
x'_2 = 2(5)+0(25)=10
$$

$$
x'_3 = 0(5)+3(25)=75
$$

$$
x'_4 = 0(5)+4(25)=100
$$

Result:

$$
x' =
\begin{bmatrix}
5 \\
10 \\
75 \\
100
\end{bmatrix}
$$

Dimension flow:

```text
2 values
   ↓
  Kᵀ
   ↓
4 values
```

Information has been expanded.



# Key Insight

Normal convolution:

$$
y = Kx
$$

Dimension flow:

```text
Large
 ↓
Small
```



Transpose convolution:

$$
x' = K^T y
$$

Dimension flow:

```text
Small
 ↓
Large
```



# Why Is It Called "Transpose Convolution"?

It is NOT because we transpose the image.

It is because:

$$
\text{Convolution} = K
$$

and

$$
\text{Transpose Convolution} = K^T
$$

where:

$$
K^T
$$

is the transpose of the matrix representing convolution.

---

# CNN Interpretation

Normal Convolution:

```text
Feature Map
      ↓
Convolution
      ↓
Smaller Feature Map
```



Transpose Convolution:

```text
Feature Map
      ↓
Transpose Convolution
      ↓
Larger Feature Map
```

---

# Final Intuition

Convolution:

```text
Many inputs
 ↓
Combine information
 ↓
Few outputs
```

Transpose Convolution:

```text
Few inputs
 ↓
Spread information
 ↓
Many outputs
```

Both use trainable kernels and both are learned through:

```text
Forward Pass
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient Update
```

The only major difference is:

```text
Convolution          → Downsampling / Compression
Transpose Convolution → Upsampling / Expansion
```


![[Screenshot 2026-06-12 213126.png]]


![[Screenshot 2026-06-12 213151.png]]

![[Screenshot 2026-06-12 213222 1.png]]

## Classification + Localization
So in the image, in addition to what category it belongs to we also want to know where it is in the image....
![[Screenshot 2026-06-12 213414.png]]

**TREAT LOCALIZATION PROBLEM AS A REGRESSION PROBLEM**


![[Screenshot 2026-06-12 213500.png]]

Goal:
Predict both the object class and its location.

Network Outputs:

1. Class Scores
   - Cat
   - Dog
   - Car
   - ...

2. Bounding Box
   - x
   - y
   - w
   - h

Training:

Classification Branch:
→ Softmax Loss

Localization Branch:
→ L2 Loss

Total Loss:

Loss = Softmax Loss + L2 Loss

Key Idea:
One CNN simultaneously learns
"What is the object?"
and
"Where is the object?"

This is an example of Multi-Task Learning.

```
Input Image
       ↓
      CNN
       ↓
 Feature Vector
       ↓

 ┌─────────────┬─────────────┐
 │             │
 ▼             ▼

 Classifier   Box Regressor

 Softmax      (x,y,w,h)

 │             │
 ▼             ▼

Softmax Loss + L2 Loss

       │
       ▼

 Total Loss
       │
       ▼

 Backpropagation
```

# Why Softmax for Classification and L2 Loss for Localization?

## Rule of Thumb

```text
Classification
→ Predict Category
→ Softmax Loss

Regression
→ Predict Numbers
→ L2 Loss
```

# Why Softmax for Classification?

Suppose the classifier outputs:
```text
Cat: 2.5
Dog: 1.2
Car: 0.3
```

These are called:
```text
Logits (Raw Scores)
```

We actually want:
```text
Probability of Cat?
Probability of Dog?
Probability of Car?
```

Softmax converts:
```text
[2.5, 1.2, 0.3]
```

into:
```text
[0.70, 0.19, 0.11]
```

Now:
```text
0.70 + 0.19 + 0.11 = 1
```

Properties:
```text
• Outputs become probabilities
• Probabilities sum to 1
• Encourages one class to dominate
```

This makes sense because:
```text
An image cannot simultaneously be:

70% Cat
and
80% Dog
```

Classification is fundamentally:
```text
Choose one category
from many possible categories.
```

Softmax is specifically designed for this task.

# Why Not Use L2 Loss for Classification?

Suppose:
```text
Ground Truth = Cat
```

One-hot encoding:
```text
[1, 0, 0]
```

Prediction:
```text
[0.7, 0.2, 0.1]
```

L2 Loss would compute:
```text
(1 - 0.7)²
+
(0 - 0.2)²
+
(0 - 0.1)²
```
This technically works.

However:
```text
Classification is not really about
numerical distance.

It is about determining
which class is correct.
```

Softmax Loss directly optimizes:
```text
Probability of the correct class.
```

Therefore it is much better suited for classification tasks.



# Why L2 Loss for Bounding Boxes?

Bounding box output:
```text
(x, y, w, h)
```

Example:
```text
(100, 50, 40, 30)
```

Ground Truth:
```text
(120, 60, 50, 35)
```

Now ask:
```text
Is x a category?
```

No.
```text
It is just a number.
```

Ask:
```text
Is width a category?
```

No.
```text
It is just a number.
```

This is therefore a:
```text
Regression Problem
```

because we are predicting continuous numerical values.
We want:
```text
Predicted Coordinates
≈
True Coordinates
```

L2 naturally measures:
```text
How far apart
two numerical values are.
```

Example:

Prediction:
```text
x = 100
```

Truth:
```text
x = 120
```

Error:
```text
20 pixels
```

L2 Loss:
```text
20² = 400
```

Large errors are penalized heavily.


# Think About The Nature Of The Output

## Classification

Output:
```text
Cat
Dog
Car
Bird
```

Nature:
```text
Discrete Categories
```

Use:
```text
Softmax Loss
```

## Bounding Box Regression

Output:
```text
x = 120.4
y = 87.1
w = 45.2
h = 31.8
```

Nature:
```text
Continuous Numerical Values
```

Use:
```text
Regression Loss
```

Examples:
```text
L2 Loss
Smooth L1 Loss
Huber Loss
```

# Why Not Use Softmax For Bounding Boxes?

Suppose:
```text
x-coordinate = 120
```

Softmax would treat:
```text
0
1
2
3
...
500
```
as separate classes.

This would create:
```text
501 classes
```
for x alone.

Then:
```text
501 classes for y
501 classes for width
501 classes for height
```

This is absurd because:
```text
Coordinates are measurements,
not categories.
```

Bounding box prediction is therefore naturally formulated as a regression problem.

---

# Mental Model

Whenever you see a network output, ask:

```text
Am I predicting a category?
```

If YES:

```text
Use Softmax Loss
```

---

Ask:

```text
Am I predicting a number?
```

If YES:

```text
Use Regression Loss

Examples:
• L2 Loss
• Smooth L1 Loss
• Huber Loss
```

---
## ASIDE: Human Pose Estimation

![[Screenshot 2026-06-12 215050.png]]

![[Screenshot 2026-06-12 215113.png]]

## Object Detection

![[Screenshot 2026-06-12 215139.png]]

![[Screenshot 2026-06-12 215222.png]]

![[Screenshot 2026-06-12 215307.png]]

![[Screenshot 2026-06-12 215328.png]]

![[Screenshot 2026-06-12 215349.png]]

![[Screenshot 2026-06-12 215436.png]]

![[Screenshot 2026-06-12 215509.png]]

# Evolution of Object Detection 

## The Goal

Image Classification answers:

```text
What is in the image?
```

Example:

```text
Cat
```

Object Detection must answer:

```text
What is in the image?
AND
Where is it?
```

Example:

```text
Cat → Bounding Box
Dog → Bounding Box
Car → Bounding Box
```

# Approach 1: Object Detection as Regression

## Idea

Treat object detection as a direct regression problem.

Input:

```text
Image
```

Output:

```text
Class
+
Bounding Box Coordinates
```

Example:

```text
Image
 ↓
CNN
 ↓

Cat

(x, y, w, h)
```

where:

```text
x = center x-coordinate
y = center y-coordinate
w = width
h = height
```

## Why It Seems Attractive

Very simple pipeline:

```text
Image
 ↓
CNN
 ↓
Class + Box
```

Only one forward pass.

End-to-end trainable.

## Major Problem

What if there are multiple objects?

Example:

```text
Dog
Cat
Car
```

Which box should the network predict?

```text
Dog box?
Cat box?
Car box?
```

The output representation becomes ambiguous.

The network does not know:

```text
How many objects exist?
```

Regression works reasonably for:

```text
Single Object Localization
```

but fails for:

```text
General Object Detection
```

with multiple objects.

# Approach 2: Object Detection as Classification

## Key Observation

CNNs are already very good at classification.

Instead of asking:

```text
Where are all objects?
```

ask:

```text
Does this region contain an object?
```

## Sliding Window Approach

Move a window across the image.

Example:

```text
┌───┐
│   │
└───┘
```

Check:

```text
Window 1
 ↓
CNN
 ↓
Cat?
```

Then:

```text
Window 2
 ↓
CNN
 ↓
Cat?
```

Then:

```text
Window 3
 ↓
CNN
 ↓
Cat?
```

Continue across the entire image.

## Flow

```text
Image
 ↓

Sliding Window

 ↓

Crop Window
 ↓
CNN
 ↓
Classification

 ↓

Move Window
 ↓
CNN
 ↓
Classification

 ↓

Move Window
 ↓
CNN
 ↓
Classification
```

## Why It Works

CNN only performs classification.

A problem CNNs already solve well.

No need to predict coordinates directly.

## Major Problems

### Problem 1: Scale

Objects appear at different sizes.

Example:

```text
Small Cat
Large Cat
```

Need windows of many sizes:

```text
32×32
64×64
128×128
256×256
```

### Problem 2: Location

Need to scan:

```text
Top Left
Top Center
Top Right
...
```

thousands of positions.

### Problem 3: Computation

For every location and scale:

```text
Crop
 ↓
CNN
```

Potentially:

```text
Millions of windows
```

per image.

This becomes computationally impossible.

## Key Insight

Most windows are useless.

Example:

```text
Sky
Grass
Road
Wall
```

contain no object.

Why waste CNN computation on them?

# Approach 3: Region Proposals

## Main Idea

Instead of checking:

```text
Every Possible Window
```

generate:

```text
Likely Object Regions
```

first.

## Selective Search

A separate algorithm analyzes:

```text
Color
Texture
Edges
Similarity
```

and proposes:

```text
~2000 Regions
```

that might contain objects.

## Important Detail

Selective Search does NOT classify.

It only says:

```text
There may be an object here.
```

Example:

```text
Proposal 1
Proposal 2
Proposal 3
...
Proposal 2000
```

## Why This Is Better

Instead of:

```text
Millions of Windows
```

we now have:

```text
~2000 Candidate Regions
```

Much smaller search space.

# Birth of R-CNN

Now combine:

```text
Region Proposals
+
CNN Classification
```

Pipeline:

```text
Image
 ↓
Selective Search
 ↓
~2000 Region Proposals
 ↓
Crop Proposal
 ↓
CNN
 ↓
Feature Vector
 ↓
Classifier
 ↓
Object Class
```

## Why R-CNN Was Revolutionary

Before:

```text
Millions of Sliding Windows
```

After:

```text
~2000 Intelligent Proposals
```

Huge reduction in search space.

CNNs could finally be used effectively for object detection.

## Remaining Problem

Still expensive.

For each proposal:

```text
Proposal
 ↓
CNN
```

If there are:

```text
2000 proposals
```

then:

```text
2000 CNN forward passes
```

must be performed.

This is extremely slow.

# Motivation For Future Models

Researchers then asked:

```text
Why run the CNN 2000 times
on almost the same image?
```

This question leads to:

```text
R-CNN
 ↓
Fast R-CNN
 ↓
Faster R-CNN
 ↓
Mask R-CNN
```

Each generation removes more redundant computation.

# Evolution Summary

```text
Object Detection as Regression
 ↓
Fails for multiple objects

Object Detection as Classification
(Sliding Window)
 ↓
Computationally too expensive

Region Proposals
 ↓
Reduce search space

R-CNN
 ↓
Region Proposals + CNN

Problem:
Still requires ~2000 CNN evaluations

Leads to:
Fast R-CNN
Faster R-CNN
Mask R-CNN
```