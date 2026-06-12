## What is really going on inside ConvNets?

# First HIDDEN Layer

- Until now we are only looking towards Convnets as a blackbox of sorts, we are saying there is an input image then it goes though multiple layers in between and then they give us class scores or bounding boxes.

Now let us look what actually is entailed in these middle layer look for and what they do?

![[Screenshot 2026-06-12 225230.png]]

For example if we look at the first layer of AlexNet consists a number of convolutional filters, each convolutional filter has shape 3x11x11 and these convolution filters get slid over the input image.

We take inner products between some chunk of the image and the weights of the convolutional filter and that gives us the output after that first convolutional layer.

In AlexNet we have 64 of these filters, but in the first layer because we are taking a direct inner product between the weights of the convolutional layer and the the pixels of the image, we can get some sense for what these filters are looking for by simply visualizing the learned weights of these filters as images themselves.

**For each of those 11x11x3 filters in AlexNet we can just visualize the filter as a little 11x11x3 image with three channels representing (R, G, B values) and then because there are 64 of these filters we just visualize 64 little 11x11 images**

Similarly for the other networks also as shown in the image above

So by looking at the images we can tell what these filters are looking for?
- A lot of these can be seen looking for oriented edges, like bars of light and dark at various angles and various positions
- We can see opposing colors like green and pink

#### Why does visualizing the weights of the filters tell you what the filter is looking for ?

## Step 1: What Is A Filter Trying To Do?

Suppose a filter is:

$$  
W =  
\begin{bmatrix}  
1 & 1 & 1 \  
0 & 0 & 0 \  
-1 & -1 & -1  
\end{bmatrix}  
$$

and it slides across an image.

For every image patch, it computes:

$$  
\text{Score} = W \cdot X  
$$

where:

```text
W = Filter Weights
X = Image Patch
```

This operation is simply a dot product.

## Step 2: When Does The Filter Give A Large Score?

A dot product becomes large when:

```text
Input Pattern
≈
Filter Pattern
```

In other words:

```text
The more similar X is to W,
the larger the score becomes.
```

Example:

Input patch:

```text
1  1  1
0  0  0
-1 -1 -1
```

looks very similar to the filter:

$$  
W =  
\begin{bmatrix}  
1 & 1 & 1 \  
0 & 0 & 0 \  
-1 & -1 & -1  
\end{bmatrix}  
$$

Therefore:

```text
Dot Product
=
Large Positive Value
```

Now consider:

```text
-1 -1 -1
 0  0  0
 1  1  1
```

This is the opposite pattern.

Therefore:

```text
Dot Product
=
Large Negative Value
```

This means the filter fires strongly whenever it sees:

```text
Bright Region
-------------
Dark Region
```

which corresponds to a:

```text
Horizontal Edge
```

## Step 3: What Do The Weights Represent?

The weights literally describe:

```text
What pattern would maximize
the filter response?
```

Consider:

```text
+2 +2 +2
-1 -1 -1
-1 -1 -1
```

This filter is effectively saying:

```text
I like bright pixels on top.

I dislike bright pixels below.
```

Therefore:

```text
This filter searches for
a horizontal edge.
```

## Why Visualization Works

Suppose we visualize the filter weights.

Convention:

```text
White = Positive Weights

Black = Negative Weights
```

A visualization may look like:

```text
■■■■■■
■■■■■■
□□□□□□
```

where:

```text
Top = White

Bottom = Black
```

Immediately we can conclude:

```text
This filter prefers:

Bright-on-top

Dark-on-bottom
```

which corresponds to:

```text
Horizontal Edge Detector
```

## Cat Analogy

Suppose after training a CNN on cats, one filter looks like:

```text
Dark
Bright
Dark
```

vertically.

You might recognize:

```text
Cat whisker pattern
```

or

```text
Vertical edge pattern
```

because the network discovered that such patterns help classify cats.

## Why Weight Visualization Works Best In Early Layers

The first convolutional layer directly sees raw pixels.

Therefore its filters often learn:

```text
Edges

Corners

Color Blobs

Textures
```

These patterns are easy for humans to interpret.

## Why Later Layers Become Hard To Interpret

Later layers no longer see pixels directly.

Instead they see:

```text
Outputs of previous filters.
```

As a result they begin detecting:

```text
Combinations of edges

Parts of eyes

Parts of noses

Wheel-like structures

Object parts

Face-like patterns
```

Eventually they may respond to:

```text
Cat-face-like features

Dog-face-like features

Car-like structures
```

The weights become much harder to interpret visually.

## The Deep Intuition

Remember:

$$  
\text{Activation} = W \cdot X + b  
$$

To obtain a large activation:

```text
Input X must resemble W.
```

Therefore:

```text
Visualizing W

≈

Visualizing the pattern
that the neuron wants to see.
```

This is the entire reason weight visualization works.

## One-Line Summary

```text
A filter produces a high response when the input patch resembles its weights.

Therefore, visualizing the filter weights reveals the pattern that most strongly activates the filter, which tells us what feature the filter is looking for.
```

# Middle Layer

![[Screenshot 2026-06-12 231008.png]]

When you go to the middle layers it is not so interesting....
you can't really visualize these images and interpret them properly you could guess is all

# Last HIDDEN Layer
Another way to try and understand what is happening in ConvNets is to try and understand the happenings at the last layer...

![[Screenshot 2026-06-12 231453.png]]

One thing that helps in this is the nearest neighbor approach.

![[Screenshot 2026-06-12 232825.png]]

This slide demonstrates one of the most important insights about Convolutional Neural Networks (CNNs): **the final hidden layer of a trained CNN learns a meaningful feature representation of images.** Traditionally, nearest-neighbor classification compares images directly in pixel space, where similarity is measured using raw pixel values. However, this often fails because two images belonging to the same class may have very different pixel arrangements due to changes in pose, lighting, background, orientation, scale, or viewpoint. For example, two dogs photographed from different angles may appear very different at the pixel level even though humans immediately recognize them as the same type of object. The left side of the slide recalls this problem by showing nearest-neighbor retrieval in pixel space, where the retrieved images often look visually similar in terms of colors and textures but may not be semantically related.

The main experiment shown in this slide is to take a trained CNN (such as AlexNet), pass an image through all convolutional and fully connected layers, and extract the **4096-dimensional feature vector** from the last hidden layer (highlighted in red). This vector can be thought of as a compact numerical summary of everything the network has learned about the image. Instead of comparing raw pixels, the authors compare these high-level feature vectors using L2 distance. The results shown in the center of the slide demonstrate that images that are close together in this learned feature space belong to the same semantic category. For example, a flower retrieves other flowers, an elephant retrieves other elephants, a ship retrieves other ships, and a dog retrieves other dogs, even though the retrieved images may differ significantly in color, pose, lighting, or background. This indicates that the CNN has learned to ignore irrelevant visual variations and preserve the features that truly define object identity.

The key takeaway is that the last hidden layer acts as a powerful feature extractor. Rather than learning explicit rules such as "this shape is a flower" or "this shape is an elephant," the network automatically learns a feature space where semantically similar objects cluster together. In this feature space, images of the same object category are located near one another, while images from different categories are far apart. This demonstrates that CNNs are not merely memorizing training images; they are learning meaningful visual representations that capture high-level concepts. This idea became foundational to modern computer vision because these learned feature vectors can be reused for image retrieval, clustering, transfer learning, similarity search, object recognition, and many other downstream tasks. In essence, the slide shows that the final hidden layer of a CNN functions as a learned embedding space in which semantic similarity emerges naturally from the training process.

**Through the standard supervised learning procedure for training classification networks there is nothing in the loss encouraging particular features to be close together IT IS JUST KIND OF A HAPPY ACCIDENT THAT THEY DO.**

**However sometimes people do train networks using something called either contrastive loss or a triplet loss which actually explicitly makes assumptions and constraints on the network such that those last their features end up having some metric space interpretation.**

Another thing that helps here is Dimensionality Reduction....

![[Screenshot 2026-06-12 234147.png]]

- *PRINCIPAL COMPONENT ANALYSIS IS ONE WAY TO DO IT*
- There is another really powerful algorithm t-SNE standing for **t - distributed stochastic neighbor embeddings** which is a non linear dimensionality reduction method.
- Here in the above image what is happening is that there is a dataset with handwritten images of numbers 1 to 9 and when its passed through a ConvNet then you get a 96x96 vector which is then reduced to 2d by using t-SNE and you can see these kind of clustering of digits according to their identity.

![[Screenshot 2026-06-12 234713.png]]

Here what happens in the above image is similar to what we discussed in the previous image a large dataset of images in imagenet is passed through a ConvNet and then the final 4096 dimensional feature vector is recorded from the last layer of each of those images

This slide demonstrates that the final hidden layer of a CNN learns a high-dimensional feature representation in which semantically similar images naturally cluster together. Every image passed through the network is converted into a **4096-dimensional feature vector** (highlighted in red on the AlexNet architecture). Each image can therefore be viewed as a single point in a 4096-dimensional space, where the coordinates correspond to learned features rather than raw pixel values. However, humans cannot visualize 4096 dimensions directly, so a dimensionality reduction technique called **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is used to project these 4096-dimensional vectors down to two dimensions while attempting to preserve local neighborhood relationships.

The left side of the slide shows a collection of original images from the dataset. The right side shows the same images after being transformed into 4096-dimensional feature vectors by the CNN and then projected into 2D using t-SNE. Remarkably, even though the network was trained only for image classification, images belonging to similar semantic categories automatically organize themselves into clusters. For example, animals tend to appear near other animals, vehicles near other vehicles, natural scenes near other natural scenes, and objects sharing similar high-level visual characteristics become neighbors in the embedding space. This indicates that the CNN has learned a meaningful internal representation of visual concepts rather than merely memorizing pixel patterns.

The key insight of the slide is that the final hidden layer acts as a learned embedding space where semantic similarity corresponds to geometric proximity. Images that are close together in this feature space tend to represent similar objects, while images that are far apart tend to belong to different categories. Dimensionality reduction is used only for visualization; the actual network still operates in the original 4096-dimensional feature space. The fact that meaningful clusters emerge after projection provides strong evidence that deep neural networks automatically learn structured representations of the world. This learned feature space becomes useful not only for classification but also for image retrieval, similarity search, clustering, transfer learning, anomaly detection, and many other downstream computer vision tasks. In essence, the slide provides visual proof that the CNN's last hidden layer has learned a rich semantic representation of images, where the geometry of the feature space reflects the underlying relationships between object categories.

**We have the pixels of the image, we have the 4096th dimensional vector then we use t-SNE to convert the 4096 dimensional vector into a 2d coordinate and then we take the original pixels of the image and place that at the 2d coordinate  corresponding to the dimensionality reduced version of the 4096 dimensional feature.

You can do the same for upper layers as well


