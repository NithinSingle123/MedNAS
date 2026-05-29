
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

