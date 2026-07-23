
## Fully connected Layer

![[Excalidraw/fullconnectlayereg.excalidraw]]

![[Excalidraw/convolutionlayer.excalidraw]]

So the main difference between this and the fully connected layer is that here we want to preserve spatial structure

## Filters

Filters are always gonna extend to the full depth of the input volume and so they are going to be just a smaller spatial area (5x5) but they are always gonna go through the full depth.

![[Excalidraw/convolayer1.excalidraw]]

## How are we going to move this filter ?

first we are going to start at the upper left hand corner and center the filter on top of every pixel in this input volume and in every position we are going to do dot product and this will produce one value in our output in activation map.

## Consider a second filter

![[Excalidraw/convolayer2.excalidraw]]

For example if we had 2 5x5 filters, we will get 6 separate activation maps and if there are 6 then 6 activation maps, we stack these up to get a new image of size 28x28x6.

## Convolutional Network (ConvNet)

A ConvNet is going to be a sequence of these convolutional layers stacked on top of each other, same way as what we had with the simple linear layers in their neural network. And then we are going to intersperse these with activation functions, so for example, a ReLU activation function.

![[convnet.excalidraw]]

## Hierarchy of filters

![[Excalidraw/filterhierarchy.excalidraw|12000]]

![[Excalidraw/convnetmodel.excalidraw]]

This is how a CNN would look like, we are going to pass it through the above layers linear, non linear and pooling layer.

![[Excalidraw/outputsize.excalidraw]]

## Output size: (N - F) / stride + 1 

e.g. N=7, F=3:
stride 1 --> (7-3)/1+1 = 5
stride 2 --> (7-3)/2+1 = 3
stride 3 --> (7-3)/3+1 = 2.33 :\

## In practice: Common to zero pad the border

![[Excalidraw/zeropad.excalidraw]]

so here when we previously took stride as 3 the output could be asymmetric but now to tackle that limitation we padded the boundary with zero and now if we look at this and stride through it with stride 3 with our 3x3 filter then we can definitely accommodate.

Some of the common filter sizes you come across is F = 3, 5, 7
if 3 zero pad with 1
if 5 zero pad with 2
if 7 zero pad with 3

We do this because we dont want the image shrinking each iteration and also losing some features along the edges


### Example

Input volume: 32x32x3
10 5x5 filters with stride 1, pad 2

Output volume size:
(36-5)/1+1 = 32 = 32x32x10

No.of parameters in this layer:

(5x5x3+1)10 = 760
here the 1 is the bias term that exists


# Convolution Layer Summary

A convolution layer:

- Accepts an input volume of size

$$
W_1 \times H_1 \times D_1
$$

---

## Hyperparameters

A convolution layer requires four hyperparameters:

- Number of filters:

$$
K
$$

- Filter spatial extent (kernel size):

$$
F
$$

- Stride:

$$
S
$$

- Zero-padding:

$$
P
$$

---

## Output Volume

The convolution layer produces an output volume of size:

$$
W_2 \times H_2 \times D_2
$$

where:

$$
W_2=\frac{W_1-F+2P}{S}+1
$$

$$
H_2=\frac{H_1-F+2P}{S}+1
$$

(width and height are computed identically by symmetry)

$$
D_2=K
$$

---

## Number of Parameters

With parameter sharing:

Each filter contains

$$
F \times F \times D_1
$$

weights.

Therefore, the total number of weights is:

$$
(F \times F \times D_1)\times K
$$

and the layer contains:

$$
K
$$

bias terms.

---

## Output Depth Slices

In the output volume:

- The \(d^{th}\) depth slice has size:

$$
W_2 \times H_2
$$

- It is produced by convolving the \(d^{th}\) filter over the input volume using stride:

$$
S
$$

- Then adding the corresponding bias term:

$$
b_d
$$

---

## Key Takeaways

### Input

$$
W_1 \times H_1 \times D_1
$$

### Hyperparameters

$$
K,\;F,\;S,\;P
$$

### Output

$$
W_2 \times H_2 \times K
$$

### Parameter Count

$$
(F \times F \times D_1)\times K + K
$$

(weights + biases)

---


# Example: CONV layer in Torch
# Spatial Convolution

```lua
module = nn.SpatialConvolution(
    nInputPlane,
    nOutputPlane,
    kW,
    kH,
    [dW],
    [dH],
    [padW],
    [padH]
)
```

Applies a 2D convolution over an input image composed of several input planes.

The input tensor in `forward(input)` is expected to be a 3D tensor:

```text
nInputPlane × height × width
```

## Parameters

- `nInputPlane`
  - Number of expected input planes in the image given to `forward()`

- `nOutputPlane`
  - Number of output planes the convolution layer will produce

- `kW`
  - Kernel width of the convolution

- `kH`
  - Kernel height of the convolution

- `dW`
  - Step of the convolution in the width dimension
  - Default: `1`

- `dH`
  - Step of the convolution in the height dimension
  - Default: `1`

- `padW`
  - Additional zeros added per width to the input planes
  - Default: `0`
  - Common choice:

```text
(kW - 1) / 2
```

- `padH`
  - Additional zeros added per height to the input planes
  - Default: `0`
  - Common choice:

```text
(kH - 1) / 2
```

Note that depending on the size of your kernel, several of the last columns or rows of the input image might be lost. It is up to the user to add proper padding.

If the input image is a 3D tensor:

```text
nInputPlane × height × width
```

the output image size will be:

```text
nOutputPlane × oheight × owidth
```

where

$$
owidth=
\left\lfloor
\frac{width+2\times padW-kW}{dW}+1
\right\rfloor
$$

$$
oheight=
\left\lfloor
\frac{height+2\times padH-kH}{dH}+1
\right\rfloor
$$


# CONV layer in Caffe

```python
layer {  
name: "conv1"  
type: "Convolution"  
bottom: "data"  
top: "conv1"  
  
# learning rate and decay multipliers for the filters  
param { lr_mult: 1 decay_mult: 1 }  
  
# learning rate and decay multipliers for the biases  
param { lr_mult: 2 decay_mult: 0 }  
  
convolution_param {  
	num_output: 96 # learn 96 filters  
	kernel_size: 11 # each filter is 11x11  
	stride: 4 # step 4 pixels between each filter application  
  
	weight_filler {  
		type: "gaussian" # initialize the filters from a Gaussian  
		std: 0.01 # distribution with stdev 0.01 (default mean: 0)  
}  
  
	bias_filler {  
		type: "constant" # initialize the biases to zero (0)  
		value: 0  
}  
}  
}
```

## Pooling Layer

they make the representations smaller and more manageable. And operates over each activation map independently.

Meaning it downsamples a large image to a smaller one

Note: this doesnt do anything in the depth we are only downsampling spatially and the input depth still remains same as output depth.

### One such method is Max Pooling

![[Excalidraw/maxpooling.excalidraw|12000]]

It is typical in pooling to have a stride such that it does not overlap.

We choose max pooling commonly because it gives a higher value

# Pooling Layer Summary

- Accepts an input volume of size:

$$
W_1 \times H_1 \times D_1
$$

---

## Hyperparameters

A pooling layer requires two hyperparameters:

- Spatial extent:

$$
F
$$

- Stride:

$$
S
$$

---

## Output Volume

Produces an output volume of size:

$$
W_2 \times H_2 \times D_2
$$

where:

$$
W_2 = \frac{W_1 - F}{S} + 1
$$

$$
H_2 = \frac{H_1 - F}{S} + 1
$$

$$
D_2 = D_1
$$

---

## Parameters

Pooling layers introduce:

$$
0
$$

trainable parameters because they compute a fixed function of the input.

---

## Notes

- It is not common to use zero-padding for pooling layers.
- Pooling reduces spatial dimensions while preserving depth.
- Common pooling operations:
  - Max Pooling
  - Average Pooling
---

Typically people dont use zero padding here because it is just downsampling

## Fully Connected Layer

# Fully Connected (FC) Layer

## Definition

A Fully Connected Layer (Dense Layer) is a neural network layer in which every neuron is connected to every neuron in the previous layer.

Unlike convolution layers, FC layers do not preserve spatial information.

---

## Mathematical Representation

Given:

$$
x \in \mathbb{R}^{n}
$$

Input vector:

$$
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

Weight matrix:

$$
W \in \mathbb{R}^{m \times n}
$$

Bias vector:

$$
b \in \mathbb{R}^{m}
$$

Output:

$$
y = Wx + b
$$

After applying an activation:

$$
a = \sigma(Wx+b)
$$

---

## Intuition

Each output neuron:

- Sees the entire input.
- Learns a weighted combination of all input features.
- Produces a higher-level representation.

---

## Example

Suppose:

$$
x =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

and

$$
W=
\begin{bmatrix}
0.1 & 0.5 & 0.2 \\
0.3 & 0.7 & 0.4
\end{bmatrix}
$$

Then:

$$
y = Wx
$$

produces:

$$
y=
\begin{bmatrix}
1.7 \\
2.9
\end{bmatrix}
$$

---

## Why CNNs Use FC Layers

Convolution layers learn:

- edges
- textures
- shapes
- object parts

The Fully Connected Layer then uses those extracted features to make the final decision.

Example:

```text
Image
↓
Convolution Layers
↓
Feature Maps
↓
Flatten
↓
Fully Connected Layer
↓
Class Scores
```

---

## Flattening

Before entering an FC layer, feature maps are usually flattened.

Example:

Feature map:

$$
8 \times 8 \times 64
$$

becomes:

$$
4096
$$

dimensional vector.

---

## Number of Parameters

For an FC layer:

Input neurons:

$$
N
$$

Output neurons:

$$
M
$$

Total weights:

$$
N \times M
$$

Total biases:

$$
M
$$

Total parameters:

$$
N \times M + M
$$

---

## Advantages

- Powerful representation learning.
- Uses information from the entire feature vector.
- Effective for final classification.

---

## Disadvantages

- Large number of parameters.
- High memory consumption.
- Can easily overfit.
- Ignores spatial structure.

---

## CNN vs Fully Connected Layer

| Convolution Layer | Fully Connected Layer |
|------------------|----------------------|
| Local connections | Global connections |
| Parameter sharing | No parameter sharing |
| Preserves spatial information | Loses spatial information |
| Few parameters | Many parameters |
| Feature extraction | Classification |

---

## NAS-Relevant Insight

In Neural Architecture Search, FC layers can be optimized by varying:

- Number of FC layers
- Hidden dimensions
- Activation functions
- Dropout rates

Many modern architectures reduce or completely remove large FC layers because they contribute a significant portion of the total parameter count.
