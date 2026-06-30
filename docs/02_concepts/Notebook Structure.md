# FOUNDATION 0 — WHY AI EXISTS

### What is Intelligence?

- What does it mean to learn?
    
- What does it mean to generalize?
    
- What is intelligence?
    

### What is Artificial Intelligence?

- What is AI trying to achieve?
    
- Why do we build intelligent systems?
    

### What is Machine Learning?

- How is ML different from traditional programming?
    
- Why does ML work?
    

### Types of Learning

- What is Supervised Learning?
    
- What is Unsupervised Learning?
    
- What is Self-Supervised Learning?
    
- What is Reinforcement Learning?
    
- When should each be used?
    

---

# FOUNDATION 1 — NEURAL NETWORK MENTAL MODEL

### What is a Neural Network?

- What problem does it solve?
    
- Why do neural networks work?
    

### Representation Learning

- What is a feature?
    
- What is a representation?
    
- Why are representations important?
    

### The Learning Loop

- What is the Forward Pass?
    
- What is a Prediction?
    
- What is Loss?
    
- What is Backpropagation?
    
- What is Weight Update?
    
- How does learning happen?
    

### Neural Network Pipeline

```text
Input
↓
Feature Extraction
↓
Representation Learning
↓
Decision Making
↓
Loss
↓
Backpropagation
↓
Weight Update
↓
Repeat
```

---

# FOUNDATION 2 — ESSENTIAL MATHEMATICS

### Scalars

- What is a scalar?
    

### Vectors

- What is a vector?
    
- Why are vectors useful?
    

### Matrices

- What is a matrix?
    
- Why are matrices useful?
    

### Tensors

- What is a tensor?
    
- Why are tensors important in deep learning?
    

### Dot Product

- What does a dot product mean?
    
- Why is it important?
    

### Matrix Multiplication

- Why do neural networks use matrix multiplication?
    
- How do shapes transform?
    

### Transpose

- Why do we transpose matrices?
    

### Gradient

- What is a gradient?
    
- Why does learning depend on gradients?
    

### Chain Rule

- Why is the chain rule important?
    
- How does it enable backpropagation?
    

---

# FOUNDATION 3 — PROBABILITY & STATISTICS

### Probability

- What is probability?
    
- How is uncertainty represented?
    

### Conditional Probability

- What is conditional probability?
    
- Why does context matter?
    

### Joint Probability

- What is joint probability?
    

### Independence

- What does it mean for events to be independent?
    

### Expectation

- What is expected value?
    
- Why is it useful?
    

### Variance

- What is variance?
    
- Why do we care about variance?
    

### Probability Distributions

#### Gaussian Distribution

- What is a Gaussian?
    
- Why is it everywhere?
    

#### Bernoulli Distribution

- What problems does it model?
    

#### Categorical Distribution

- How does it model multiple classes?
    

---

# FOUNDATION 4 — OPTIMIZATION

### Objective Function

- What are we trying to optimize?
    

### Loss Function

- What is loss?
    
- Why do we need it?
    

### Gradient Descent

- How does gradient descent work?
    
- Why does it find good solutions?
    

### Learning Rate

- What happens if it is too large?
    
- What happens if it is too small?
    

### Local Minima

- What are local minima?
    

### Saddle Points

- What are saddle points?
    
- Why are they problematic?
    

---

# FOUNDATION 5 — CORE NEURAL NETWORK COMPONENTS

### Affine Layer

- What is an affine transformation?
    
- Why do we use XW + b?
    

### Activation Functions

#### Sigmoid

- Why was it used?
    

#### Tanh

- How is it different from sigmoid?
    

#### ReLU

- Why did ReLU become dominant?
    

#### Leaky ReLU

- What problem does it solve?
    

### Softmax

- Why convert scores into probabilities?
    

### Cross Entropy

- Why is it used for classification?
    

---

# FOUNDATION 6 — UNDERSTANDING TRAINING

### Underfitting

- What causes underfitting?
    

### Overfitting

- What causes overfitting?
    

### Generalization

- What does it mean to generalize?
    

### Dataset Splits

#### Training Set

#### Validation Set

#### Test Set

- Why do we need all three?
    

### Bias-Variance Tradeoff

- What is bias?
    
- What is variance?
    
- Why is balancing them important?
    

---

# FOUNDATION 7 — THE ARCHITECTURE LENS

### What is an Architecture?

- What makes something an architecture?
    

### What Makes an Architecture Good?

- Accuracy
    
- Efficiency
    
- Memory
    
- Latency
    
- Robustness
    
- Interpretability
    

### Why New Architectures Keep Appearing

- Why aren't CNNs enough?
    
- Why were RNNs created?
    
- Why did Transformers emerge?
    
- Why do we need efficient models?
    
- Why does NAS exist?
    

---

# FOUNDATION 8 — RESEARCH MINDSET

### How to Read a Research Paper

- What sections matter most?
    

### How to Evaluate an Idea

- What makes an idea valuable?
    

### How to Ask Questions

- How do researchers identify problems?
    

### What Makes a Research Contribution?

- What counts as novelty?
    
- What counts as impact?
    

---



# AFTER FOUNDATIONS

Begin CS231n traversal in order:

```text
Linear Classification
↓
Neural Networks
↓
Backpropagation
↓
Optimization
↓
CNNs
↓
Detection & Localization
↓
RNNs
↓
Attention
↓
Generative Models
↓
Reinforcement Learning
↓
Efficient Deep Learning
↓
Adversarial Examples & Robustness
```

This structure will give you a complete "AI from first principles → architectures → research" progression before you ever reach NAS, Transformers, or future research papers.



PART 0 — FOUNDATIONS

Foundation 0 — Why AI Exists
Foundation 1 — Neural Network Mental Model
Foundation 2 — Essential Mathematics
Foundation 3 — Probability & Statistics
Foundation 4 — Optimization
Foundation 8 — Research Mindset

──────────────────────────────

PART 1 — CS231n Journey

(Lecture-wise / Concept-wise Notes)

──────────────────────────────

PART 2 — Architecture Foundations

Foundation 6 — Understanding Neural Network Architectures

Foundation 7 — Architecture Design Principles

──────────────────────────────

PART 3 — Architecture Journey

LeNet
AlexNet
VGG
GoogLeNet
ResNet
DenseNet
...
Transformers
NAS