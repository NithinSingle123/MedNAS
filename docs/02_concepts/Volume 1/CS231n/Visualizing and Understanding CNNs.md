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


## Visualizing Activations

Here we can also visualize activations of intermediary layers to try and understand what is going on...

![[Pasted image 20260613043130.png]]

so here you can see a convolution run through a web cam to try and identify what is infront of it. In this case it is the face of the person sitting infront of the camera.

- Now you will notice here that there is nothing significant in the grid and its all noisy but there is one highlighted visualized activation which seems to be trying to light up and identify the face of the person sitting infront of the camera.

**Are the black activations dead ReLUs??**
And the answer is no, dead ReLUs are something that are out of use for the whole network but in this case they are not dead they are just not active for the given input.

## Maximally Activating Patches

![[Screenshot 2026-06-13 043811.png]]

Maximally Activating Patches

The idea behind maximal activation visualization is to understand what a particular neuron or feature map inside a CNN has learned to detect. Instead of visualizing the filter weights directly, we observe which image regions cause the strongest activation of that neuron.

Suppose we choose a particular layer in the network, for example Conv5. If Conv5 contains 128 feature maps of size 13 × 13, we may select a specific channel, such as channel 17. This channel represents one learned feature detector.

Next, thousands of images are passed through the trained network. For every image, the activation values produced by channel 17 are recorded. Since the feature map is 13 × 13, each location corresponds to a specific receptive field in the original image. Some locations will produce higher activations than others.

The patches of the original images that generate the largest activation values are then extracted and visualized. These extracted regions are called maximally activating patches because they correspond to inputs that most strongly excite the selected feature detector.

If the selected feature map has learned to detect eyes, then the maximally activating patches will mostly contain eyes from different animals and objects. If it has learned wheel-like structures, then the highest-activation patches will contain wheels, circular objects, camera lenses, and other similar patterns. If it has learned text-like features, the patches will contain letters, signs, logos, and printed text.

The images shown on the right side of the slide are examples of maximally activating patches. Each grid corresponds to a different neuron or feature map. Notice that although the patches come from different images, many of them share a common visual pattern. One neuron appears to respond strongly to eyes, another to circular objects such as wheels and camera lenses, another to text and letters, and another to animal faces. This indicates that the network has automatically learned specialized detectors without being explicitly programmed.

This method is powerful because it provides evidence of what a neuron actually responds to in practice. Weight visualization tells us what pattern the filter prefers mathematically, while maximal activation visualization shows real-world image regions that trigger the neuron most strongly. Since deeper layers learn increasingly complex concepts, this technique is often much easier to interpret than visualizing the raw filter weights.

The overall procedure is:

1. Select a layer and a feature map (channel).
2. Pass many images through the network.
3. Record activation values of the chosen channel.
4. Find the locations with the highest activations.
5. Map those activations back to the corresponding image patches.
6. Visualize the patches.
7. Infer the visual concept that the neuron has learned.

This technique demonstrates that deep CNNs gradually evolve from detecting simple edges and textures in early layers to detecting object parts, faces, eyes, wheels, text, and other high-level semantic concepts in deeper layers.

The key insight of this slide is:

```
Weight Visualization:
"What pattern does the filter mathematically want?"

Maximally Activating Patches:
"What real image regions actually make the filter fire?"
```

## Occlusion Experiments

![[Screenshot 2026-06-13 044409.png]]

- So what we wanna do here is figure out which parts of the input image caused the network to make its classification decision. So what we will do is take our input image and then we will block out some part of that image and just replace it with the mean pixel value from the dataset.
- So the idea here is that when plot a heat graph for the scores, if for a case of occlusion in the image the scores change drastically then that part was really important for the classification decision.

## Saliency Map

![[Screenshot 2026-06-13 045121.png]]

Saliency Maps are a visualization technique used to identify which pixels in an input image most influence a CNN's prediction.

The idea is to compute the gradient of the class score with respect to every input pixel. Pixels with larger gradient magnitudes are considered more important because small changes in those pixels would significantly affect the prediction.

Procedure:
1. Pass an image through the network.
2. Select the score of the predicted class (or any target class).
3. Compute the gradient of that score with respect to the input image.
4. Visualize the magnitude of the gradients as a heatmap.

Bright regions in the saliency map indicate areas that the network considers most important for making its decision.

Unlike maximal activation patches, which show what a neuron generally responds to across many images, saliency maps explain why the network made a specific prediction for a particular image.

![[Screenshot 2026-06-13 045455.png]]

- can be used for semantic segmentation

![[Screenshot 2026-06-13 045716.png]]

## Intermediate Features via (guided) backprop

![[Screenshot 2026-06-13 045935.png]]

- What happens is that here you tweak the process of backpropagation and change you backpropagate through the non linearites
- So in this case while backpropagating you are supposed let only positive values through the non linearities and ignore negative (only keeping track of positive influences)

Doing this makes the images we get more highlighted and with more clarity as to which parts of the image affect the score more...

![[Screenshot 2026-06-13 050313.png]]

## Gradient Ascent

![[Screenshot 2026-06-13 050526.png]]

Saliency Maps vs Gradient Ascent

Saliency maps ask:
"Given a real image, which pixels caused the neuron to activate?"

Gradient ascent asks:
"Forget real images. What image would I have to create to make this neuron activate as much as possible?"

Think about it this way.
Suppose you have a neuron that you suspect detects dog faces.
How do you verify it?

One way is to show it thousands of images and see which ones activate it the most.

Another way is to ask the neuron directly:
"Show me your dream image."

That is exactly what gradient ascent does.

 #### Step 1: Start with Random Noise 

Imagine creating an image like:

▓▒░▓▒░▓  
░▓▒░▓▒░  
▒░▓▒░▓▒

This is pure random garbage.
Feed this image into the CNN.
Suppose the chosen neuron produces an activation value:

f(I) = 5

#### Step 2: Compute the Gradient

Calculate:

∂f/∂I

This gradient tells us:
- Which pixels should become brighter
- Which pixels should become darker

to increase the neuron's activation.

#### Step 3: Modify the Image

During normal training we update the weights:

Weights ← Weights − η∇Loss

In gradient ascent we freeze all network weights and instead update the image itself:

Image ← Image + η∇f(I)

This is called Gradient Ascent because we are maximizing the neuron's activation rather than minimizing a loss.

Repeat Many Times

Random Noise  
↓  
Slight Pattern  
↓  
Recognizable Shape  
↓  
Neuron's Preferred Image

After hundreds of iterations the image gradually evolves into a pattern that strongly excites the neuron.

Eventually we obtain an image that makes the neuron extremely happy.

What is f(I)?

f(I) represents the activation value of the chosen neuron when image I is fed into the network.

Higher value of f(I) means:
- Stronger neuron response
- Greater neuron activation

Therefore our objective is:

Maximize f(I)

Why Do We Need R(I)?

If we only maximize f(I), the optimizer quickly discovers strange tricks.

The resulting image often becomes:
- High-frequency noise
- Random textures
- Unnatural patterns

that humans cannot interpret.

The neuron may love the image, but it looks like nonsense to us.

For example, TV static may activate a neuron more strongly than a real dog image.

To prevent this, we introduce a regularizer:

R(I)

The role of R(I) is to:

- Reduce noise
- Encourage smoothness
- Produce more natural-looking images
- Prevent optimization from exploiting unrealistic patterns

Intuition of the Full Objective

I* = argmax_I (f(I) + R(I))

Read this as:

"Find the image I* that simultaneously:
1. Maximally activates the neuron
2. Still looks reasonably natural"

Deep Intuition
Imagine interviewing a neuron.

Saliency Maps ask:
"Why did you like THIS image?"

Gradient Ascent asks:
"If you could design your own image, what would it look like?"

The generated image is essentially:
- The neuron's imagination
- The neuron's dream image
- The neuron's ideal input

By studying this image we gain insight into the visual concept that the neuron has learned.

Summary

Saliency Maps:  
Given a real image, identify which pixels influenced the prediction.

Gradient Ascent:  
Generate a synthetic image that maximally activates a neuron.

Saliency Maps explain:  
"Why did the neuron respond?"

Gradient Ascent explains:  
"What does the neuron want to see?"

This is why gradient ascent is one of the most powerful techniques for visualizing and understanding the internal representations learned by deep CNNs.

**The question here might arise how the hell does one change the image altogether instead of weights??**

The trick is:
```
In gradient ascent,the image itself becomes the variable.
```

Normally in training:
```
Image (fixed)      
↓
CNN      
↓
Prediction      
↓
LossUpdate Weights
```

The image never changes.

In gradient ascent:
```
Image (variable)      
↓
CNN (fixed)      
↓
Neuron Activation

Update Image
```

The weights are frozen.
The image becomes the thing being optimized.

Let's use a tiny example.

Suppose the image is just:
```
I =[100  50]
   [ 20 200]
```

Pixel values.
Feed it into the network.

Suppose the neuron activation is:
```
f(I) = 5
```

Now compute:
```
∂f/∂I
```

Suppose we get:
```
[+3  -2]
[+1  -4]
```

This means:
```
Pixel (1,1):Increase it
Pixel (1,2):Decrease it
Pixel (2,1):Increase it slightly
Pixel (2,2):Decrease it strongly
```

Update the image:
```
Image ← Image + η∂f/∂I
```

If:
```
η = 1
```

then:
```
Old Image[100  50]
         [ 20 200]
         
         +

Gradient[+3  -2]
		[+1  -4]

		 =

New Image[103  48]
		 [ 21 196]
```

The image has literally changed.

Feed the new image again:

```
New Image      
↓
CNN      
↓
f(I) = 8
```

Activation increased.
Good.
Repeat.

**The reason this works is because, in math an image is just a big vector of numbers.**

![[Pasted image 20260613052708.png]]

![[Pasted image 20260613052742.png]]

![[Pasted image 20260613052837.png]]

![[Pasted image 20260613052920.png]]

## Fooling Images/ Adversarial Examples

![[Pasted image 20260613053543.png]]

![[Pasted image 20260613053510.png]]

- So what happens here is that for example we take an example of an elephant ask the network to modify the image to represent it under some other class like a koala bear. Now some might thing during this process the elephant would start morphing and then sprout cute ears at the end.
- But in reality what happens is the same image is classified as an image.

## Deep Dream: Amplify existing features

![[Screenshot 2026-06-15 220148.png]]

```python
def objective_L2(dat):
	dat.diff[:] = dat.data
	
def make_step(not, step_size=1.5, end = 'inception_4c/output', jitter=32, clip=True, objective=objective_L2):
	'''Basic gradient ascent step.'''
	
	src = net.blobe['data'] # input image is stored in Nets's 'data' blob
	dat = net.blobe[end]
	
	ox, oy = np.random.randint(-jitter, jitter+1, 2)
	src.data[0] = np.roll(np.roll(arc.data[0], ox, -1), oy, -2) # apply jitter shift
	
	net.forward(end=end)
	objective(dat) # specify the optimization objective
	net.backward(start=end)
	g = src.diff[0]
	# apply normalized ascent step to the input image # L1 Normalize gradients
	src.data[:] += step_size/np.abs(g).mean() * g
	
	src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
	
	if clip:
		bias = net.transfomrer.mean['data']
		arc.data[:] = np.clip(src.data, -bias, 255-bias)
```

Code is very simple but it uses couple of tricks:
1. Jittering image before you compute your gradients
	- So here what happens here is, rather than running the exact image through the network instead you will shift the image over by two pixels and kind of wrap the other two pixels over here. And this is a kind of regularize to prevent each of these, it regularizes a little bit to encourage a little bit of extra special smoothness in the image

2. L1 Normalize gradients
	- Its a kind of a useful trick sometimes when doing the image generation problems

3. Clipping the pixel values once in a while
	- images should actually be between zero to 2.55, so this is a kind of projected gradients descent where we project on to the space of actual valid images. But now when we do all this and start with the image of a sky then the end product gives you dream like figures appearing all over the sky..

![[Pasted image 20260615233228.png]]

**This tells us that the model was trained on more categories of dogs**

![[Pasted image 20260615233420.png]]

This is a lower level interpretation of deep dream in its lower layers trying to find edges or corners and stuff like that through those swirls...

**IF YOU RUN THIS THING FOR A LONG TIME AND ADD IN SO MULTISCALE PROCESSING YOU GET SOME REALLY INTERESTING AND CRAZY IMAGES**

![[Pasted image 20260615233703.png]]

The code for deep dream is online by google you can make your own versions

## Feature Inversion

- gives a sense what types of elements of image are captured in different layers

![[Pasted image 20260615234103.png]]

![[Pasted image 20260615234207.png]]


## Texture Synthesis

- Here the idea is that we are give an input of patch of texture, we want to build some model and generate a larger piece of that same texture.

![[Pasted image 20260615234416.png]]

# COME BACK TO THIS LATER

