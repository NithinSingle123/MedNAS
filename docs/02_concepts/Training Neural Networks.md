
So we saw at any particular layer how we have the data coming in, we multiply by weights and pass it through an activation function.

![[Excalidraw/actfn.excalidraw|150000]]

# Activation Functions


## I. Sigmoid   $$\sigma(x)=\frac{1}{1+e^{-x}}$$

```desmos-graph
y=1/(1+\exp(-x))
```

- Squashes numbers to range [0, 1]
- historically popular since they have nice interpretation as a saturating "firing rate" of a neuron

### Problems

### 1. Saturated neurons "kill" the gradients

![[Excalidraw/sigmoid.excalidraw]]

when x = -10 (very negative) then the output will be close to 0 meaning that the gradient becomes 0 and we are going to get very small gradient passing backwards.

when x=0 it will be in the regime or range 

when x=10 (large positive) the region is flat and it is going to kill the gradient


### 2. Sigmoid outputs are not zero - centered


![[Excalidraw/actfn.excalidraw|1000]]

Consider what happens when the input to a neuron is always positive, all are x's are going to be positive and they are gonna be multiplied to some weights 

then the updates are either all positive for the positive case or all negative for the negative case
this is why you also mean 0 mean data

![[Excalidraw/sigmoid2.excalidraw|150000]]

So we need to have zero meaned input so that we have positive and negative values both

### 3. exp() is a bit compute expensive

the exponential function is expensive not compared to all the calculations you are doing but a minor insight


## II. tanh(x)   $$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

```desmos-graph
y=\tanh(x)
```

- Squashes numbers to range [-1, 1]
- zero centered
- still kills gradients when saturated

it is better than sigmoid but still has its drawbacks


## III. ReLU (Rectified Linear Unit) f(x) = max(0, x)

- Does not saturate (in + region)
- Very computationally efficient
- Converges much faster than sigmoid/tanh in practice (e.g. 6x)
- Actually more biologically plausible

Note: if your input is negative then it is put to zero and if it is positive its going to pass through the identity

### Problems

### 1. Not zero - centered
### 2. Annoyance bcz if x is less than or equal to zero then the gradient will become zero and its only stable in its positive half

![[Excalidraw/relu.excalidraw|150000]]

the data cloud is our training data
if the weights are off the input then there will be dead relu
also if the learning rate is high then the weights are updated at a large difference and as a result the relu gets knocked off meaning after you start learning in between they go bad and then dead

It is a research problem but it does fine on the training data for now

people like to initialize the ReLU neurons with slightly positive bias like 0.01 to take care of the problem


## IV. Leaky ReLU $$\text{LeakyReLU}(x)=\begin{cases}x & x \ge 0\\0.01 x & x < 0\end{cases}$$

![[Excalidraw/leakyrelu.excalidraw|15000]]


- Does not saturate 
- Computationally efficient
- Converges much faster than sigmoid/tanh in practice
- will not die !!


## IV. Parametric ReLU $$\text{LeakyReLU}(x)=\begin{cases}x & x \ge 0\\\alpha x & x < 0\end{cases}$$
leaky relu is just a spl case of parametric relu where sigma is 0.01

### And then there is also ELU exponential Linear Units where the equation is as follows

$$\text{LeakyReLU}(x)=\begin{cases}x & x \ge 0\\\alpha (exp(x)-1) & x < 0\end{cases}$$
![[Excalidraw/exprelu.excalidraw]]

- all benefits of ReLU
- closer to zero mean outputs
- negative saturation regime compared to with Leaky ReLU adds some robustness to noise

but as it involves exp() it requires computation

# Maxout Neuron

# $$max(w_1^Tx+b_1, w_2^Tx+b_2)$$



- Does not have the basic form of dot product -----> non linearity
- Generalizes ReLU and Leaky ReLU 
- Linear Regime! Does not saturate! Does not die!

However it doubles the number of parameters/neurons

# Points to note and remember:-

1. Use ReLU. Be careful with your learning rates
2. Try out Leaky ReLU / Maxout / ELU
3. Try out tanh but dont expect much
4. Don't use sigmoid

---

# Data Preprocessing

![[Pasted image 20260604011114.png]]

we want to zero center the data as we dont want all of them to be positive or negative altogether
we should normalize the data corresponding to the std deviation to ensure all are in same range and contribute equally.

# Weight Initialization

