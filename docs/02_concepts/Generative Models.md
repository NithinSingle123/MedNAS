
# Supervised vs Unsupervised Learning

### Supervised Learning

- Data: (x, y) where x is data, y is label
- Goal: Learn a function to map x ---> y
- Examples: Classification, regression, object detection, semantic segmentation, image captioning, etc..

### Unsupervised Learning

- Data: x where just data, no labels!
- Goal: Learn some underlying hidden structure of the data
- Examples: clustering, dimensionality reduction, feature learning, density estimation, etc...

Advantage is that in unsupervised learning there are no labels so the training data is cheap..

## Generative Models

- Given training data, generate new samples from same distribution
- Training data ~ $$ P_{data}(x)$$
- Generated samples ~ $$ P_{model}(x)$$
- Want to learn p_model(x) similar to p_data(x)

- Addresses density estimation, a core problem in unsupervised learning

##### Density Estimation

Density Estimation is considered one of the fundamental goals of unsupervised learning because, unlike supervised learning, we do not have labels.

Instead of learning:

x → y

we try to learn how the data itself is distributed, or mathematically, the probability distribution of the data:

p(x)

### Intuition

Suppose you have thousands of face images.

In supervised learning you might have labels such as Person A, Person B, and Person C.

In unsupervised learning you only have the images themselves:

Image 1  
Image 2  
Image 3  
...

No labels are available.

The question becomes:

What does the data look like?

Density estimation tries to answer questions such as:

- Which images are common?
- Which images are rare?
- Which images are likely to belong to the data distribution?

### Example

```
Training Data (Heights):160
165
170
172
168
171
169

Density Estimation Learns:
Most people are around 170 cm
Few people are around 120 cm
Few people are around 250 cm
Graphically:          
           
Probability

           ^
           |           
           |      /\           
           |     /  \           
           |    /    \
           |___/______\_______          
                 Height
```

The peak corresponds to a high-density region where many samples occur.

### What does "Density" Mean?

Density refers to how much data exists around a particular point.

High density means many samples are located nearby.

Low density means very few samples are located nearby.

### Why is it Useful?

Once we know p(x), many important machine learning tasks become possible.

**1. Generate New Data**

By learning the distribution of real-world data, models can generate realistic faces, handwriting, speech, and other content.

Examples:

- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Diffusion Models

**2. Anomaly Detection**

If p(x) is very small, then the sample is unusual or abnormal.

Examples:

- Credit card fraud
- Medical abnormalities
- Machine failures

**3. Clustering**

High-density regions often correspond to natural clusters.

For example, images of cats may form one dense region while images of dogs form another dense region, even though no labels are provided.

### Mathematical Definition

Density estimation attempts to learn:

p(x)

where:

- x = data sample
- p(x) = probability density

High value of p(x):

- Likely sample
- Frequently observed

Low value of p(x):

- Unlikely sample
- Rarely observed

### Why is it a Core Problem in Unsupervised Learning?

Because unsupervised learning has no labels.

The only information available is the data itself.

Therefore, the most natural question we can ask is:

Can we learn how the data is distributed?

Learning this distribution is exactly the goal of density estimation.

Once the distribution is known, many other unsupervised learning tasks such as generation, anomaly detection, clustering, and representation learning become possible.


## Several Flavors
- Explicit density estimation: explicitly define and solve for p_model(x)
- Implicit density estimation: learn model that can sample from p_model(x) without explicitly defining it.

## **Why Generative Models**

Applications:
- Realistic samples for artwork, super-resolution, colorization, etc..
- Generative models of time-series data can be used for simulation and planning (reinforcement learning applications!)
- Training generative models can also enable inference of latent representations that can be useful as general features.

## Taxonomy of Generative Models

![[Pasted image 20260616001854.png]]

# Pixel RNN

## Fully visible belief network

- Explicit density model
- Use chain rule to decompose likelihood of an image x into product of 1-d distributions:
## $$p(x) = \prod_{i=1}^{n} p(x_i \mid x_1, \ldots, x_{i-1})$$

where:
- p(x) is likelihood of image x
- x_i is the probability of i'th pixel value(given all previous pixel values)

Then maximaize likelihood of training data.

- Complex distribution over pixel values --> Express using a neural network

## What is a generative model trying to do?

Suppose I show you these images:
```
Dog
Dog
Dog
Dog
Dog
```
and train a model.
What do I want at the end?

Not:
```
Input → Dog
Input → Cat
```
That's classification.

Instead I want:
```
Generate a brand new dog image
```
that never existed before.

## The first question becomes

How do we mathematically define:

```
"A realistic image"
```

CS231n answers:
```
Let's learn the probability of images.
```

In notation:
```
p(x)
```

where:
```
x = entire image
```

## Why can't we directly learn p(x)?

Imagine a tiny image:
```
32 × 32 × 3
```

That's:
```
3072 pixel values
```

So:
```
p(x)
```

really means:
```
p(x_1,x_2,x_3,\ldots,x_{3072})
```
a gigantic joint probability distribution.
Impossible to model directly.

## Then comes the chain rule

Probability theory gives a trick.

Instead of learning:
```
p(x_1,x_2,\ldots,x_n)
```

we can rewrite it as:
```
p(x)=\prod_{i=1}^{n} p(x_i|x_1,\ldots,x_{i-1})
```

This is **not a neural network idea**.
This is just a probability identity.

## Tiny example

Suppose an image has only 3 pixels:
```
x1
x2
x3
```

Then:

```
p(x_1,x_2,x_3)
=
p(x_1)
p(x_2|x_1)
p(x_3|x_1,x_2)
```
Same probability.
Just rewritten.

## Why is this useful?

Because now the model only has to answer:
```
Given previous pixels,what should the next pixel be?
```

instead of:
```
Describe every possible image on Earth.
```

## PixelRNN intuition

Imagine drawing an image.
You draw:
```
Pixel 1
```

Then ask:
```
What should Pixel 2 be?
```

Then:
```
What should Pixel 3 be?
```
and so on.
That's literally PixelRNNs

## Why the weird visualization?

Those dots are usually showing:
```
Current pixel
       ↑
depends on
all previous pixels
```

Example:
```
● → ● → ● → ● → ●
```
Like an RNN over pixels.

The network predicts:
```
Next pixel value
```
given all previous ones.

## What does training do?

The lecture says:
> Maximize likelihood of training data.

Meaning:
If a real image is:
```
Dog image
```

the model should assign:
```
p(x)
```
a large value.

For nonsense:
```
Random static noise
```

the model should assign:
```
p(x)
```
a small value.

![[Pasted image 20260617011233.png]]


# Pixel CNN

![[Pasted image 20260617011600.png]]

![[Pasted image 20260617011908.png]]

![[Pasted image 20260617012139.png]]

![[Pasted image 20260617012235.png]]


# Variational autoencoders

### Autoencoders

They are an unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data.

![[Excalidraw/encoder.excalidraw]]

```
Encoder:
   |
   |
   V
Originally:
Linear + Non-Linearity (sigmoid)
Later: Deep, fully-connected
Later: ReLU CNN
```

> Here z is USUALLY smaller than x (dimensionality reduction) and this is because it only retains the important features (z should represent the good features in x)

![[Pasted image 20260617021648.png]]

![[Pasted image 20260617021817.png]]

Here there is conv and upcov distinctly mentioned because we are first downscaling the image to only its important features and then upscaling it again.

![[Pasted image 20260617022120.png]]

![[Pasted image 20260617022328.png]]

**The question here now arises, we have seen that autoencoders can reconstruct data and can learn features to initialize a supervised model
Features capture factors of variation in training data. Can we generate new images from an autoencoder?**

![[Pasted image 20260617023041.png]]

The true parameters of this generation process are $$\theta \quad and \quad \theta^* $$
So we have the parameters of our prior and our conditional distributions and what we want to do in order to have a generative model to be able to generate new data, we want to estimate these parameters of our true parameters

**How should we represent this model??**
- Choose prior p(z) to be simple, e.g. Gaussian.
- Conditional p(x|z) is complex and we have to generate an image so we represent it in the form of a neural network.

We want to be able to train the model to know the estimate of the parameters.

![[Pasted image 20260617224532.png]]

**How can we train this model??**
![[Pasted image 20260617224601.png]]

**What is the problem with this??**
This integral is not going to be tractable

![[Pasted image 20260617224806.png]]

and by bayes theorem...
![[Pasted image 20260617230101.png]]

![[Pasted image 20260617230203.png]]

![[Pasted image 20260617230253.png]]

![[Pasted image 20260617230330.png]]

- Encoder and decoder networks are also called recognition/inference and generation networks

![[Pasted image 20260617230819.png]]

![[Pasted image 20260617230941.png]]

![[Pasted image 20260617231033.png]]

### Lets look at the forward pass in this case

![[Pasted image 20260617231349.png]]

- For every minibatch of input data: compute the forward pass and then the backpropagation
![[Pasted image 20260617231607.png]]

![[Pasted image 20260617231823.png]]

![[Pasted image 20260617231858.png]]


# Generative Adversarial Networks

![[Pasted image 20260617232156.png]]

![[Pasted image 20260617232333.png]]

![[Pasted image 20260617232547.png]]

![[Pasted image 20260617232833.png]]

![[Pasted image 20260617233158.png]]

![[Pasted image 20260617233413.png]]

![[Pasted image 20260617233527.png]]

![[Pasted image 20260617233649.png]]

# REVISIT

