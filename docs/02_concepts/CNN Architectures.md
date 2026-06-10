# Alex Net

- AlexNet from 2012, was the first large scale CNN that was able to do well on the ImageNet classification task.

Architecture:
```
CONV1
MAX POOL 1
NORM 1
CONV 2
MAX POOL 2
NORM 2
CONV 3 
CONV 4 
CONV 5
MAX POOL 3
FC 6
FC 7 
FC 8
```

![[Pasted image 20260609003919.png]]

## Sizes involved in the AlexNet


### First Layer

[[Convolutional Neural Networks#Output size (N - F) / stride + 1]]

- Input: 227x227x3 images
- First layer (CONV 1): 96 11x11 filters applied at stride 4 so the output volume would be [55x55x96] (check the referred notes to find more)

[[Convolutional Neural Networks#Number of Parameters]]

- The number of parameters is (11x11x3)x96 = 35000 and the additional multiplication by a factor of 3 is because of the input depth being 3 and the filter depth is also three bcz of it.

### Second Layer

- Input: 227x227x3 images
- After CONV 1: 55x55x96
[[Convolutional Neural Networks#Pooling Layer Summary]]

- Second Layer (POOL1): 3x3 filters applied at stride 2
- Output volume: 27x27x96
- The number of parameters here are 0

![[Screenshot 2026-06-09 020857.png]]

# VGGNet
- Deeper networks and smaller filters
- 8 layers (AlexNet) --> 16 to 19 layers (VGG16Net)
- Only 3x3 CONV stride1, pad 1 and 2x2 MAX POOL stride 2

![[Screenshot 2026-06-09 224540.png]]

**Why use smaller filters ??**
- we take small filters we have fewer parameters and we try and stack more of them instead of having larger filters have smaller filters with more depth instead.

**What is the effective receptive field of three 3x3 conv (stride 1) layers?**
- The thing to remember her is that these layers overlap each other here
Now look at the intuition behind effective receptive field of three 3x3 conv layers

![[Excalidraw/reception.excalidraw|10000]]
So the effective receptive field here is 7x7 which is equivalent to one 7x7 conv layer.
[[Convolutional Neural Networks#Filters]]

- So what happens here is that it is deeper and has more non linearities
- And fewer parameters: $$ 3 * (3^2 C^2) \quad vs. \quad 7^2 C^2 \quad  for \quad C \quad channels \quad  per \quad  layer $$
![[Pasted image 20260609230650.png]]

- just go through these layers same as we did with AlexNet.
- Total memory: 24M * $ bytes ~= 96MB / image (only forward! ~`*`2 for bwd )
- Total parameters: 138M parameters
- this is going to be about 100mb for image and this is pretty heavy in terms memory usage

- Here the points to note are that:
1. Most memory is in early CONV
2. And most parameters are in late FC

And the above stack of VGG16 can be viewed like this as well:
```
Softmax
fc8
fc7
fc6
Pool
conv 5-3
conv 5-2
conv 5-1
Pool
conv 4-3
conv 4-2
conv 4-1
Pool
conv 3-2
conv 3-1
Pool
conv 1-2
conv 1-1
```

# GoogleNet

- Deeper networks, with computational efficiency
- 22 layers
- GoogleNet really looked at this problem of computation efficiency and tried to design a network architecture which was very efficient in the amount of compute. They did this using something called the *inception module* and stacking a lot of these inception modules on top of each other.
- There is also no fully connected layers in this network
- In total there is only 5 million parameters! which is 12x less than that of AlexNet.

![[Screenshot 2026-06-10 000457.png]]

![[Screenshot 2026-06-10 000508.png]]

## Inception Module

So the idea behind the inception module is that they wanted to design a good local network topology (think of it as a network within a network and stack a lot of these local networks on top of each other)

![[Screenshot 2026-06-10 000823.png]]

- What happens here is that they apply filter operations in parallel on input form previous layer:
	- Multiple receptive field sizes for convolution(1x1, 3x3, 5x5)
	- Pooling operation (3x3)
- And then concatenate all filter outputs together depth-wise

**Now let us look at what is the problem with this**
Computational complexity is going to be a problem here....

Let us look at an example to know how:
![[Excalidraw/eginception.excalidraw]]

Here a question can arise as to how everything thing is being dealt in 28x28s and the answer is that we are doing the zero padding here to preserve the spatial dimensions so that we can later concatenate them depth-wise.

So the image the input 28x28x256 is the output of the previous inception model and 28x28x672 is going to be the input of the next inception model.

Here if you notice:
```
Conv Ops:
[1x1 conv, 128] 28x28x128x1x1x256
[3x3 conv, 192] 28x28x192x3x3x256
[5x5 conv, 128] 28x28x96x5x5x256

Total: 854M operations
```

And here you can see how compute expensive this shit is...
Pooling layer also preserves feature depth, which means total depth after concatenation can only grow at every layer!

**How do we deal with and how do we keep this manageable??**
We can address this by using **bottleneck layers** and try and project these feature maps to lower dimensions before our expensive layers

**For Bottleneck layers is it possible to use some other types of dimensionality reduction ??**
The answer is yes.

![[Screenshot 2026-06-10 004257 1.png]]

![[Screenshot 2026-06-10 004356.png]]

![[Screenshot 2026-06-10 004700 1.png]]

So the new inception module takes these one by one convs and adds these at a bunch of places in these modules in order to alleviate this expensive compute.

![[Screenshot 2026-06-10 010231.png]]

**What info might be lost by doing 1x1 conv in the beginning?**


![[Screenshot 2026-06-10 230658.png]]
- This is more like a vanilla conv net we have seen earlier (six sequence layers)

![[Pasted image 20260610230911.png]]
- Multiple inception modules stacked on top of each other

![[Screenshot 2026-06-10 230942.png]]
- this is showing the classifier output
- Notice that here they have removed the expensive fully connected layers and it turns out that the model works great without them.

![[Screenshot 2026-06-10 231047.png]]
- These extra stems coming out are auxiliary classification outputs, so in simple words they are mini networks with an average pooling, a 1x1 conv, a couple of fully connected layers here going to the softmax and also a 1000 way softmax with the ImageNet classes.
- And so you are actually using your ImageNet training classification loss in three separate places here. The standard end of the network, as well as in these two places earlier on in this network.

- There is 22 total layers with weights (including each parallel layer in an inception module)

**Why do we have to inject gradients in earlier layers using the auxiliary classification outputs??**
The problem is that when you have very deep networks and you are going all the way back through these, some of the gradient signal can become minimized so we are just strengthening it in between providing some additional signal.


# ResNet

- Very deep networks using residual connections
- 152-layer model for ImageNet

## What happens when we continue stacking deeper layers on a "plain" convolutional neural network ?

![[Screenshot 2026-06-10 234049.png]]

- And to a common question as whether we can get better results by just implementing deeper layers, the answer is no
- Above there are two graphs showing the comparison between the 20-layer and 56-layer networks in the aspect of training error and test error.
- You will see that in the case of test error, the 56 layer network is doing worse than the 20 layer network, so the deeper layer was not able to do better.
- But in the case of training error, in the case of 56 layer network, you think that I have a really deep network and I have tons of parameters maybe if its starting to overfit at some point
- But what actually happens is that when you are overfitting, you would expect to have very good, very low training error and just bad test error, but what's happening here is that in the training error the 56 layer network is also doing worse than the 20 layer network
- *Even though the deeper network performs worse, this is not caused by overfitting*

**Here the hypothesis made by the ResNet group is that the problem that is being faces is an optimiztion problem, deeper models are harder to optimize.**

### And then a question might arise, WHY?


### How do we solve this?
**The deeper model should be able to perform at least as well as the shallower model.
A solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping.**

![[Screenshot 2026-06-10 235526.png]]

Let us look at what's really happening here,...

## Ordinary CNN Block
```
Input Feature Map
        x
        │
        ▼
     Conv
        │
        ▼
     ReLU
        │
        ▼
     Conv
        │
        ▼
Output Feature Map
```
- Only one path exists and everything must go through the convolutions

## Residual Block
```
               ┌──────────────┐
               │              │
               │              ▼
Input x ───────┤          Identity
               │         (Copy x)
               │
               ▼
            Conv
               │
               ▼
            ReLU
               │
               ▼
            Conv
               │
               ▼
             F(x)
               │
               ▼
         Add x + F(x)
               │
               ▼
      Output Feature Map
```

Now what x as an input here represents is that, it is a feature map of the input.
- Now if you see everything in the previous base CNN had to go through the convolutions, but here what we are doing is that we are keeping a copy of x stored without undergoing any convolutions.
- Now the output you get after performing the convolutions and applying ReLU is what we call F(x) here. Now what the hypothesis proposes is that you add x to F(x) and the output is going to perform well.

**A question arises here as to why it performs well is addition the only way?**
So this is still in research and at that time their working on the summing way cracked a path for them and it worked well is all.

![[Screenshot 2026-06-11 023349.png]]

Note: For ResNet architectures with more than 50 layers deep, they also use bottleneck layers similar to what googleNet did to improve efficiency.

#### Training ResNet in Practice
- Batch Normalization after every CONV layer
- Xavier/2 initialization from He et al.
- SGD + Momentum (0.9)
- Learning rate: 0.1, divided by 10 when validation error plateaus
- Mini-batch size 256
- Weight decay of 1e-5
- No dropout used

![[Pasted image 20260611024205.png]]







