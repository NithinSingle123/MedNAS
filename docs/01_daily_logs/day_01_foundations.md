# MedNAS — Day 1 Foundations

**Date:** 27-05-2026  
**Phase:** Phase 1 — NAS Foundations  

## Primary Goal
Understand CNN fundamentals and build the first CIFAR-10 training pipeline.

---

# 1. Concepts Learned

## Concept 1 — Convolution Layers

### Topics Covered
- Filters/Kernels
- Feature Extraction
- Channels
- Padding
- Stride
- Feature Maps

### Key Understanding
Convolution layers detect spatial patterns in images by sliding learnable kernels across image regions.

### Important Insight
CNNs preserve spatial relationships unlike fully connected layers.

### Resource Used
- CS231n CNN Lecture
- [Paste YouTube/video/article link]

### Notes
[Write your own understanding here]

---

## ## Concept 2 — Forward Pass

### Topics Covered

- Affine (Fully Connected) Layers
    
- Matrix Multiplication
    
- ReLU Activation
    
- Score Computation
    
- Softmax Probabilities
    

### Key Understanding

The forward pass transforms input data into class scores through a sequence of mathematical operations. Each layer extracts increasingly useful representations until the final layer produces scores for each class.

### Resource Used

- Stanford CS231n Lecture 4
    
- Stanford CS231n Assignment 1
    

### Notes

Implemented the complete forward pass of a two-layer neural network from scratch using NumPy.

Architecture:

Input  
→ Affine Layer (W1, b1)  
→ ReLU  
→ Affine Layer (W2, b2)  
→ Scores  
→ Softmax  
→ Probabilities

Learned how matrix multiplication propagates information through layers and how ReLU introduces non-linearity.

---

## Concept 3 — Backpropagation

### Topics Covered

- Computational Graphs
    
- Chain Rule
    
- Gradient Flow
    
- Affine Layer Backward Pass
    
- ReLU Backward Pass
    
- Softmax + Cross Entropy Derivative
    
- Numerical Gradient Checking
    
- SGD Updates
    

### Key Understanding

Backpropagation computes gradients of the loss with respect to every parameter in the network by repeatedly applying the chain rule backwards through the computational graph.

### Resource Used

- Stanford CS231n Lecture 4
    
- Stanford CS231n Assignment 1
    

### Notes

Learned that gradients multiply along a path and add across multiple paths.

Key result:

# $$  
dscores

# \frac{\partial L}{\partial scores}

probs - \text{one-hot}(y)  
$$

Successfully implemented:

- dW1
    
- db1
    
- dW2
    
- db2
    
- ReLU backward
    
- Softmax backward
    
- L2 Regularization gradients
    

Verified correctness through numerical gradient checking with relative errors on the order of 1e-9.

---

## 2 Layer Neural Network Implementation

### Architecture

Input  
→ Affine Layer (W1, b1)  
→ ReLU  
→ Affine Layer (W2, b2)  
→ Softmax

### Tasks Completed

- Forward pass implementation
    
- Softmax loss implementation
    
- L2 regularization
    
- Full backpropagation implementation
    
- Numerical gradient checking
    
- Mini-batch sampling
    
- SGD parameter updates
    
- Prediction function implementation
    

### Key Debugging Milestones

- Fixed forward pass returning None
    
- Implemented Softmax probability computation
    
- Understood derivation of dscores
    
- Fixed regularization mismatch causing gradient check failures
    
- Implemented SGD updates correctly
    

### Final Results

Gradient Check Results:

W1 max relative error: 3.56e-09

b1 max relative error: 2.74e-09

W2 max relative error: 3.44e-09

b2 max relative error: 4.45e-11

Training loss reduced from approximately 1.25 to 0.017, confirming successful learning.

---

## Important Thoughts

Deeper CNNs likely learn better representations because successive layers build increasingly abstract features from lower-level patterns.

NAS can potentially optimize architectural choices such as kernel size, channel count, number of layers, pooling strategies, and hidden dimensions while balancing accuracy, parameter count, and training cost.

---

## What I Understood Well

- Forward propagation
    
- ReLU activations
    
- Chain rule intuition
    
- Gradient flow through computational graphs
    
- Softmax loss
    
- SGD parameter updates
    
- Gradient checking
    

## What Confused Me

- Jacobians versus practical backpropagation
    
- Why gradients add at branching nodes
    
- Derivation of Softmax + Cross Entropy gradients
    
- Relationship between regularization loss and regularization gradients
    

## Biggest Technical Insight Today

Backpropagation is not magic. It is simply repeated application of the chain rule through a computational graph. Once dscores is computed, the remaining gradients systematically propagate through the network.

## Biggest Mistake Today

Initially implemented the regularization loss incorrectly, which caused large gradient-check errors for W1 and W2. Fixing the mismatch between the regularization loss term and its derivative reduced errors to approximately 1e-9.