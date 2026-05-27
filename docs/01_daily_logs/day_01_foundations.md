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

## Concept 2 — Forward Pass

### Topics Covered
- Layer-by-layer transformation
- Activations
- Representation learning

### Key Understanding
The forward pass defines how information flows through the network.

### Resource Used
- [Paste resource link]

### Notes
[Your understanding]

---

## Concept 3 — Backpropagation

### Topics Covered
- Loss calculation
- Gradients
- Weight updates
- `optimizer.step()`

### Key Understanding
Backpropagation adjusts model weights to reduce prediction error.

### Resource Used
- [Paste resource link]

### Notes
[Your understanding]

---

# 2. Implementation Work

## CIFAR-10 Pipeline

### Tasks Completed
- Dataset loading
- DataLoader setup
- Transform setup

### Files Worked On
```text
mininas/training/dataset.py
```

### Problems Faced
Faced some syntax issue 
Issues related to the compatibility of pytorch and matplotlib

### Solutions
enabled virtual environment to deal with it and now it works fine

---

## Baseline CNN

### Architecture
- Conv2D
- ReLU
- MaxPool
- Linear Layer

### Files Worked On
```text
mininas/models/cnn_builder.py
```

### Observations
- Tensor shapes changed after pooling
- Channel depth increased representational capacity

---

## Training Loop

### Components Implemented
- Forward pass
- Loss calculation
- Backpropagation
- Optimizer step

### Files Worked On
```text
mininas/training/train.py
```

### Metrics Observed
- Training loss
- Accuracy
- Training time

---

# 3. Research Intuition Developed

## Questions Explored
- Why do deeper CNNs learn better representations?
- Why does pooling reduce computation?
- Which architecture properties could NAS optimize?

## Important Thoughts
[Write your own ideas]

---

# 4. Architecture/NAS Thinking

## NAS-Relevant Observations
- Kernel size is tunable
- Channel count affects parameters
- Pooling changes feature dimensions

## Potential Search Space Components
- Number of convolution layers
- Kernel sizes
- Channel sizes
- Pooling strategies

---

# 5. GitHub Progress

## Commits Made
- Initialized MedNAS repository
- Added project structure
- Implemented CIFAR-10 pipeline

---

# 6. Tomorrow's Objectives

- Understand configurable architectures
- Build dynamic CNN builder
- Learn architecture representation
- Explore parameter counting

---

# 7. Daily Reflection

## What I Understood Well
[Write here]

## What Confused Me
[Write here]

## Biggest Technical Insight Today
[Write here]

## Biggest Mistake Today
[Write here]