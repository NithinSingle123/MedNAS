# MedNAS — Day 2 Foundations

**Date:** [7-06-2026]

**Phase:** Phase 1 — NAS Foundations

## Primary Goal

Learn how neural architectures are represented and build a configurable CNN architecture generator that can later be searched automatically by NAS algorithms.

---

# 1. Concepts Learned

## Concept 1 — Architecture Representation

### Topics Covered

- Architecture Encoding
    
- Search Space Representation
    
- Layer Representation
    
- Hyperparameter Representation
    
- Architecture Genotypes
    

### Key Understanding

Neural architectures must be represented in a machine-readable format before NAS algorithms can search, compare, mutate, or optimize them.

### Resource Used

- NAS Survey
    
- DARTS Paper (Introduction)
    

### Notes

[Write your understanding]

---

## Concept 2 — Search Spaces

### Topics Covered

- Fixed Search Spaces
    
- Layer-wise Search Spaces
    
- Cell-Based Search Spaces
    
- Operation Choices
    
- Hyperparameter Search
    

### Key Understanding

NAS does not search all possible neural networks. It searches only within a carefully designed search space that defines which architectures are allowed.

### Resource Used

- NAS Survey
    

### Notes

[Write your understanding]

---

## Concept 3 — Parameter Counting

### Topics Covered

- Conv2D Parameters
    
- Linear Layer Parameters
    
- Total Model Size
    
- Computational Cost
    
- Model Complexity
    

### Key Understanding

Parameter count is often treated as an optimization objective alongside accuracy because larger models require more memory and computation.

### Resource Used

- CS231n
    
- PyTorch Documentation
    

### Notes

[Write your understanding]

---

# 2. Implementation Work

## Dynamic CNN Builder

### Tasks Completed

- Configurable number of convolution layers
    
- Configurable channel counts
    
- Configurable kernel sizes
    
- Configurable pooling strategy
    
- Automatic architecture generation from configuration
    

### Files Worked On

mininas/models/cnn_builder.py

### Problems Faced

[Write here]

### Solutions

[Write here]

---

## Architecture Encoding

### Tasks Completed

Create architecture representations such as:

[(Conv,32,3),  
(Conv,64,3),  
(Pool),  
(FC,128)]

Represent architectures using Python lists, dictionaries, or custom classes that can later be used by NAS algorithms.

### Files Worked On

mininas/search_space/encoding.py

### Observations

[Write here]

---

## Search Space Generator

### Tasks Completed

- Define possible channel values
    
- Define possible kernel sizes
    
- Define possible layer depths
    
- Generate candidate architectures
    

### Files Worked On

mininas/search_space/search_space.py

### Observations

[Write here]

---

## Parameter Counter

### Tasks Completed

- Count trainable parameters
    
- Display model size
    
- Compare architecture complexity
    

### Files Worked On

mininas/utils/model_stats.py

### Metrics Observed

- Total Parameters
    
- Model Size
    
- Layer-wise Parameter Distribution
    

---

# 3. Research Intuition Developed

## Questions Explored

- Which architecture parameters affect accuracy most?
    
- Why is search space design critical in NAS?
    
- How large should a search space be?
    
- What makes an architecture efficient?
    

## Important Thoughts

[Write here]

---

# 4. Architecture/NAS Thinking

## Search Space Components

- Number of Conv Layers
    
- Number of Filters
    
- Kernel Sizes
    
- Pooling Operations
    
- Activation Functions
    
- Hidden Dimensions
    

## First NAS Search Space Draft

Example:

[(Conv, choice[16,32,64], choice[3,5]),  
(Conv, choice[32,64,128], choice[3,5]),  
(Pool),  
(FC, choice[64,128,256])]

### Observations

[Write here]

---

# 5. GitHub Progress

## Commits Made

- Dynamic CNN Builder
    
- Architecture Encoding System
    
- Search Space Generator
    
- Parameter Counter
    

---

# 6. Tomorrow's Objectives

- Multi-objective optimization basics
    
- Pareto optimality
    
- Accuracy vs Model Size tradeoff
    
- Fitness functions
    
- Introduction to NSGA-II
    

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