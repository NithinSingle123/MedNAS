
# MedNAS — Day 2 Foundations

**Date:** [Fill Date]

**Phase:** Phase 1 — NAS Foundations

## Primary Goal

Learn how architectures are represented and create configurable CNN architectures that can later be searched automatically.

---

# 1. Concepts Learned

## Concept 1 — Architecture Representation

### Topics Covered

- Architecture Encoding
    
- Search Space
    
- Layer Representation
    
- Hyperparameters
    

### Key Understanding

Neural architectures must be represented in a machine-readable format before NAS can search them.

### Resource Used

- NAS Survey
    
- DARTS Paper (Introduction)
    

### Notes

[Write your understanding]

---

## Concept 2 — Search Spaces

### Topics Covered

- Fixed Search Spaces
    
- Cell-Based Search Spaces
    
- Operation Choices
    
- Hyperparameter Search
    

### Key Understanding

NAS does not search every possible network. It searches inside a predefined search space.

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
    

### Key Understanding

Model complexity is one of the objectives NAS often optimizes.

### Resource Used

- CS231n
    
- PyTorch Documentation
    

### Notes

[Write your understanding]

---

# 2. Implementation Work

## Dynamic CNN Builder

### Tasks Completed

- Configurable channels
    
- Configurable kernel sizes
    
- Configurable number of layers
    

### Files Worked On

mininas/models/cnn_builder.py

### Problems Faced

[Write here]

### Solutions

[Write here]

---

## Architecture Encoding

### Tasks Completed

Create architecture representation such as:

[(Conv,32,3),  
(Conv,64,3),  
(Pool),  
(FC,128)]

### Files Worked On

mininas/search_space/encoding.py

### Observations

[Write here]

---

## Parameter Counter

### Tasks Completed

- Count trainable parameters
    
- Display model size
    

### Files Worked On

mininas/utils/model_stats.py

### Metrics Observed

- Total Parameters
    
- Model Size
    

---

# 3. Research Intuition Developed

## Questions Explored

- Which architecture parameters matter most?
    
- Why is search space design important?
    
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
    
- Architecture Encoding
    
- Parameter Counter
    

---

# 6. Tomorrow's Objectives

- Multi-objective optimization basics
    
- Pareto optimality
    
- Accuracy vs Model Size tradeoff
    
- Fitness functions
    

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