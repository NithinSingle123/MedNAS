CS231n
↓
Dynamic CNN Builder
↓
Parameter Counting
↓
PyTorch Architecture Engineering
↓
Pareto Optimization
↓
NSGA-II
↓
NAS Survey
↓
Mini NAS Project
↓
DARTS
↓
MedNAS Prototype
↓
Healthcare Constraints
↓
Research Paper


FOUNDATIONS ✅
    │
    ▼
CS231n ✅
    │
    ▼
Notebook (ongoing)
    │
    ▼
Phase 2
PyTorch & Component Mastery
    │
    ▼
Phase 3
Architecture Mastery
    │
    ▼
Phase 4
Paper Reading
    │
    ▼
Phase 5
Architecture Reproduction
    │
    ▼
Phase 6
Architecture Improvements
    │
    ▼
Phase 7
MedNAS

---
	
# The Golden Rule

For every topic:

```
1. Learn the concept (20%)↓2. Build it (60%)↓3. Experiment with it (20%)
```

The implementation—not the lecture—is where your understanding will solidify.

---

# Resource 1 (Our Primary Resource)

## LearnPyTorch by Daniel Bourke ⭐⭐⭐⭐⭐

This is, in my opinion, the best resource for your current stage.

Why?

- Research-oriented
- Uses modern PyTorch
- Clean code
- Covers best practices
- Very implementation focused

Most importantly:

It teaches PyTorch **through building models**.

Exactly what we want.

---

# Resource 2 (Documentation)

## Official PyTorch Documentation ⭐⭐⭐⭐⭐

You'll eventually live here.

Not now.

But after every topic we'll read the official docs.

Because researchers constantly use documentation.

---

# Resource 3 (Source Code)

Eventually we'll read

```
torch.nntorch.optimtorchvision.models
```

This becomes incredibly valuable later.

Not yet.

---

# Resource 4 (Our Implementations)

This will become your biggest resource.

Every component we implement ourselves.

Eventually you'll have

```
implementations/Linear/Conv2D/BatchNorm/Adam/ResNetBlock/Attention/Transformer/
```

That folder becomes more valuable than any course.

---

# Resource 5 (Your Notebook)

This is no longer course notes.

It's becoming

> Neural Architecture Engineering

Everything you truly understand goes there.

---

# What I DO NOT want you to use

No random YouTube tutorials.

No "PyTorch in 2 hours."

No GeeksForGeeks.

No "Top 50 PyTorch Interview Questions."

No "Master PyTorch in 30 Days."

They're fine for quick lookups, but they shouldn't drive your learning.

---

# Here's how I'd organize Phase 2.

---

## Module 1

PyTorch Fundamentals

Topics:

- Tensor creation
- Shapes
- Dtypes
- Devices
- Indexing
- Broadcasting
- Matrix multiplication
- Reshape/View/Permute

Mini Project:

Build tensor operations manually.

---

## Module 2

Autograd

Topics:

- Computational graph
- requires_grad
- backward()
- Gradient accumulation
- no_grad()

Mini Project:

Implement Linear Regression.

---

## Module 3

Neural Network Module

Topics:

- nn.Module
- Parameters
- Buffers
- Forward pass

Mini Project:

Build an MLP.

---

## Module 4

Data Pipeline

Topics:

- Dataset
- DataLoader
- Custom Dataset
- Transforms

Mini Project:

Train on CIFAR-10.

---

## Module 5

Training Pipeline

Topics:

- Loss
- Optimizer
- Scheduler
- Validation
- Checkpointing

Mini Project:

Complete CNN training loop.

---

## Module 6+

Now we enter component mastery.

One component.

One implementation.

One week.

---

# Here's what I would NOT do.

Don't finish LearnPyTorch from start to finish.

Instead:

```
Learn tensors↓Implement tensors↓Experiment↓Next topic
```

Very different.

---

# Where CS231n comes back

One beautiful thing.

Everything you learned now has implementation.

Example:

CS231n

```
Wx+b
```

Phase 2

```
nn.Linear()
```

CS231n

```
Convolution
```

Phase 2

```
nn.Conv2d()
```

CS231n

```
Backpropagation
```

Phase 2

```
loss.backward()
```

CS231n

```
Optimization
```

Phase 2

```
optimizer.step()
```

See what happened?

We're translating mathematics into code.

---

# My Proposal

I want Phase 2 to become almost like a **research apprenticeship**.

Every week will have exactly this format:

```
MondayConcept↓TuesdayPyTorch implementation↓WednesdayImplement from scratch↓ThursdayExperiment↓FridayDebug↓SaturdayReflection + Notebook↓SundayMini project
```

That rhythm is sustainable and closely mirrors how researchers work.

---

# One thing I want to add

I think we're ready to stop following other people's curricula and start following **our own**.

We'll use LearnPyTorch as the reference, but **I won't say "complete Chapter 1, then Chapter 2."**

Instead, I'll act as the curriculum designer.

I'll decide the order based on one question:

> **"What knowledge gives the maximum leverage for understanding neural architectures?"**

That means we may skip sections, revisit others later, and interleave implementation with theory when it makes sense. The goal is no longer to "finish a PyTorch course." The goal is to become someone who can open the implementation of a state-of-the-art model and understand not just _what_ the code does, but _why_ it is written that way.

I genuinely think this will make Phase 2 the most enjoyable and transformative part of your journey so far.

---

Yes—with **one important change**: I would **not watch all 13 resources first and only then return**. That creates exactly the information-overload problem you wrote your article about: lots of exposure, delayed synthesis, and weak connections between families.

For the way you're building this notebook, the better loop is:

**Watch one family → come back → deepen the mathematics with me → run small PyTorch experiments → design the notebook pages → move to the next family.**

I would sequence it like this:

1. **Foundation:** CS231n Lecture 3. Then we extract the underlying language of objectives, hinge loss, Softmax/cross-entropy, regularization context, and loss landscapes.
    
2. **Regression family:** MSE → L1 → Smooth L1 → Huber → Gaussian NLL → Poisson NLL. We derive gradients, compare geometry and robustness, and connect probabilistic assumptions.
    
3. **Classification family:** Hinge → BCE → BCEWithLogits → Cross-Entropy → NLL → weighted CE → Focal Loss. This is where logits, sigmoid, softmax, likelihood, and numerical stability need to become one coherent mental model.
    
4. **Distribution family:** KL divergence and its relationship to entropy, cross-entropy, and distribution matching.
    
5. **Margin, ranking, and metric learning:** Margin Ranking → Hinge Embedding → Cosine Embedding → Contrastive → Triplet losses, plus the remaining PyTorch margin variants. Here the central idea becomes shaping representation geometry.
    
6. **Sequence alignment:** CTC. Learn the conceptual mechanism first, then the forward-backward mathematics to the depth you actually need.
    
7. **Vision and medical segmentation:** Dice → IoU/Jaccard → Tversky → Focal → Focal Tversky → boundary-aware losses. For your future MedNAS direction, this deserves more depth than generic learners usually give it.
    
8. **Architecture/deployment future sight:** knowledge distillation objectives—hard targets, soft targets, temperature, and why this interacts with compression and deployment.
    
9. **Generative future sight:** reconstruction objectives → VAE/ELBO → adversarial objectives → a conceptual view of diffusion training objectives. Enough to understand the landscape without derailing the architecture phase.
    

My recommendation is therefore: **start with CS231n Lecture 3 now. Don't take polished final notes while watching it.** Keep rough observations, equations, questions, and points that surprise you.

Then come back, and we'll start with the first family. For each family, we'll use the same process: understand why the family exists, derive the core mathematics, examine gradients and geometry, compare failure modes, run controlled PyTorch experiments, connect the objective to architecture and task design, and only then decide what deserves permanent space in your handwritten notebook.

That approach is much closer to your actual goal. You're not trying to become someone who can name many loss functions; you're trying to develop enough mathematical intuition to eventually **modify, combine, or design an objective when the standard one doesn't encode the behaviour your problem requires**.



| Order  | Family / Block                                    | Losses and ideas we will cover                                                                                   | Watch before coming back to me                                                                                                                                                                                                                                                                                  |
| ------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**  | **Foundations**                                   | Loss objectives, SVM/Hinge, Softmax, Cross-Entropy, regularization context, loss landscapes, gradient intuition  | [CS231n 2017 — Lecture 3: Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc&utm_source=chatgpt.com)                                                                                                                                                                                  |
| **2**  | **Regression**                                    | MSE, L1/MAE, Smooth L1, Huber, Gaussian NLL, Poisson NLL                                                         | **No single video I trust for the whole block. Come directly to me after Stage 1 and we'll derive this family together.**                                                                                                                                                                                       |
| **3**  | **Classification**                                | Hinge recap, BCE, BCEWithLogits, Cross-Entropy, NLL, weighted CE, Focal Loss                                     | CS231n Lecture 3 gives the base; then we'll deepen this family together rather than stacking repetitive introductory videos.                                                                                                                                                                                    |
| **4**  | **Distribution & Information**                    | Entropy, Cross-Entropy relationship, KL divergence, distribution matching                                        | We will build this mathematically together from the classification block. No separate prerequisite video needed.                                                                                                                                                                                                |
| **5**  | **Margin, Ranking & Metric Learning**             | Margin Ranking, Hinge Embedding, Cosine Embedding, Contrastive Loss, Triplet Loss, hard-negative mining, InfoNCE | [MIT 6.7960 — Similarity-Based Representation Learning](https://learn.mit.edu/video/88430?playlist=88418&utm_source=chatgpt.com), then optionally [Siamese Representation Learning lecture](https://www.youtube.com/watch?v=6x4IPITZ4dw&utm_source=chatgpt.com) for a more concrete architecture-and-loss view. |
| **6**  | **Sequence Alignment**                            | CTC intuition, blank token, valid alignments, path probabilities, forward-backward idea                          | [CTC Explained](https://www.youtube.com/watch?v=jDPl1QJGLpE&utm_source=chatgpt.com) first; if you want the deeper lecture afterward, use [CTC Full Lecture](https://www.youtube.com/watch?v=5Rj0J9AuGw0&utm_source=chatgpt.com).                                                                                |
| **7**  | **Segmentation I — Region Losses**                | Dice, IoU/Jaccard, BCE combinations, overlap objectives                                                          | [Semantic Segmentation Loss Functions](https://www.youtube.com/watch?v=NqDBvUPD9jg&utm_source=chatgpt.com) as an introduction, then we derive and compare the losses ourselves.                                                                                                                                 |
| **8**  | **Segmentation II — Imbalance & Medical Imaging** | Focal, Tversky, Focal Tversky, sensitivity/specificity trade-offs, small-lesion behaviour                        | No single video is sufficient for the depth we want. We'll work through the equations and original motivations together after Stage 7.                                                                                                                                                                          |
| **9**  | **Boundary-Aware Objectives**                     | Boundary loss concepts, region vs boundary supervision, compound objectives                                      | We'll handle this as a research-oriented reading-and-derivation block rather than forcing a weak video resource.                                                                                                                                                                                                |
| **10** | **Knowledge Distillation**                        | Hard targets, soft targets, temperature, teacher–student loss, weighted compound objectives                      | We'll choose the lecture when we reach this block; this is future-sight for deployment-aware architecture work rather than a prerequisite for CNNs.                                                                                                                                                             |
| **11** | **Generative Objectives**                         | Reconstruction loss, VAE objective/ELBO, adversarial objectives, diffusion objective intuition                   | Future-sight block only. We should choose resources when you reach it rather than expanding the current phase unnecessarily.                                                                                                                                                                                    |
## Classification Family

### Chapter 1

**What Does a Classifier Actually Output?**

(Logits, probabilities, sigmoid intuition)

---

### Chapter 2

**How Does a Network Decide Between Two Classes?**

(Binary classification)

---

### Chapter 3

**How Should Wrong Probabilities Be Penalized?**

(Binary Cross Entropy)

---

### Chapter 4

**How Do We Classify More Than Two Classes?**

(Softmax)

---

### Chapter 5

**Why Is Cross Entropy the Natural Choice?**

(Cross Entropy + NLL)

---

### Chapter 6

**What Happens When the Dataset Is Unbalanced?**

(Weighted CE, Focal Loss)

---

### Chapter 7

**Can We Compare Entire Probability Distributions Instead of Labels?**

(KL Divergence, Label Smoothing)



Classification Family
│
├── Chapter 1
│   What Does a Classifier Output?
│   ✓ Logits
│   ✓ Probabilities
│   ✓ Sigmoid motivation
│
├── Chapter 2
│   Binary Classification
│   ✓ One output neuron
│   ✓ Positive/negative class
│   ✓ Targets (0/1)
│   ✓ Why probability matters
│
├── Chapter 3
│   Deriving Binary Cross Entropy
│   ✓ Likelihood
│   ✓ Log-likelihood
│   ✓ Negative log-likelihood
│   ✓ BCE from first principles
│
├── Chapter 4
│   Multi-Class Classification
│   ✓ Softmax
│   ✓ Cross Entropy
│   ✓ NLLLoss
│
└── Later
    ✓ Focal Loss
    ✓ Label Smoothing
    ✓ KL Divergence
    ✓ Dice
    ✓ Tversky




I'm actually glad you asked this because **I want to redesign the entire Classification Family before we go any further.**

The Regression Family taught me something about how you learn.

You don't like chapters named after formulas.

You like chapters that answer **one fundamental question**.

So let's make this notebook read like a **graduate textbook**, not PyTorch documentation.

---

# Loss Functions

```
PART I
Foundations
```

- Probability
    
- Likelihood
    
- Maximum Likelihood Estimation
    
- Negative Log Likelihood
    
- Backpropagation
    
- Loss landscapes
    
- Gradient intuition
    

---

```
PART II
Regression Family ✅
```

### Chapter 1

**What Is a Regression Problem?**

- Continuous prediction
    
- Residual
    
- Error intuition
    

---

### Chapter 2

**How Should Numerical Error Be Measured?**

- MSE
    
- L1
    
- Smooth L1
    
- Huber
    

---

### Chapter 3

**Can We Predict Entire Probability Distributions?**

- Gaussian NLL
    
- Poisson NLL
    

---

Regression Family Complete.

---

# PART III

# Classification Family

This is where we are now.

---

# Chapter 1

## What Does a Classifier Actually Output?

This chapter answers

> "What comes out of a classifier?"

Contents

- Why labels cannot be predicted directly
    
- Why probabilities matter
    
- Logits
    
- Probability interpretation
    
- Design requirements of Sigmoid
    
- Why Sigmoid exists
    

---

# Chapter 2

## How Does a Network Decide Between Two Classes?

This chapter answers

> "How do binary classifiers work?"

Contents

- Binary classification
    
- Positive vs Negative class
    
- Why one output neuron is sufficient
    
- Binary targets
    
- Decision threshold
    
- Probability vs prediction
    

---

# Chapter 3

## How Should We Penalize Wrong Probabilities?

This is the chapter we just finished.

Contents

### Engineering derivation

- Why MSE is insufficient
    
- Confidence vs numerical error
    
- Desired behaviour of the loss
    
- Why logarithms naturally appear
    

↓

### Probability derivation

- Bernoulli distribution
    
- Bernoulli likelihood
    
- Maximum Likelihood Estimation
    
- Negative Log Likelihood
    
- Binary Cross Entropy derivation
    

↓

Final result

[  
L=-\left(y\log p+(1-y)\log(1-p)\right)  
]

Notice

This chapter is

**not about BCE.**

It's about

> **Why BCE must exist.**

---

# Chapter 4

This is the chapter I want to redesign.

Originally

I called it

> BCEWithLogitsLoss

I don't like that anymore.

Instead

---

# Why Do Deep Learning Libraries Never Compute the Obvious Formula?

This is one of my favourite chapters already.

Contents

- Floating-point numbers
    
- Precision limits
    
- Overflow
    
- Underflow
    
- Why log(0) is dangerous
    
- Numerical stability
    
- BCEWithLogitsLoss
    
- LogSoftmax
    
- CrossEntropyLoss internals
    
- Why libraries fuse mathematical operations
    

This chapter isn't really about BCEWithLogitsLoss.

It's about

> **Engineering mathematics.**

---

# Chapter 5

## How Does a Network Choose Among Many Classes?

Binary →

Multi-class

Contents

- Why sigmoid no longer works
    
- One-hot encoding
    
- Probability simplex
    
- Softmax intuition
    
- Softmax derivation
    

---

# Chapter 6

## How Should We Penalize Multi-Class Predictions?

Contents

- Cross Entropy
    
- NLLLoss
    
- CrossEntropyLoss
    
- Relationship between them
    
- Why PyTorch expects logits
    

One of the most satisfying equations in PyTorch

```text
CrossEntropyLoss

=

LogSoftmax

+

NLLLoss
```

---

# Chapter 7

## What Happens When Reality Is Messy?

This is the practical chapter.

Contents

- Class imbalance
    
- Weighted Cross Entropy
    
- Focal Loss
    
- Hard examples
    
- Easy examples
    
- Label smoothing
    
- Calibration
    
- KL Divergence intuition
    

Notice how the difficulty naturally increases.

---

# Then...

Once Classification finishes

we move into

```
PART IV
Information-Theoretic Losses
```

where we study

- Entropy
    
- Cross Entropy (deeper mathematical interpretation)
    
- KL Divergence
    
- Jensen-Shannon Divergence
    
- Mutual Information
    
- Distribution matching
    

This is where we connect machine learning with information theory in a much deeper way than the classification family needed.

---

# Then

```
PART V
Metric Learning
```

- Contrastive Loss
    
- Triplet Loss
    
- Margin Ranking
    
- Cosine Embedding
    
- InfoNCE
    

---

# Then

```
PART VI
Sequence Losses
```

- CTC
    
- Alignment
    
- Forward-backward algorithm
    

---

# Then

```
PART VII
Segmentation
```

- Dice
    
- IoU
    
- Tversky
    
- Boundary Loss
    
- Medical imaging losses
    

---

# Then

```
PART VIII
Modern Objectives
```

- Knowledge Distillation
    
- ELBO
    
- GAN Losses
    
- Diffusion objectives
    

---

# Why I like this structure much more

Look at the chapter titles:

> **What Does a Classifier Actually Output?**

> **How Does a Network Decide Between Two Classes?**

> **How Should We Penalize Wrong Probabilities?**

> **Why Do Deep Learning Libraries Never Compute the Obvious Formula?**

> **How Does a Network Choose Among Many Classes?**

> **How Should We Penalize Multi-Class Predictions?**

> **What Happens When Reality Is Messy?**

Notice something?

**Not a single chapter is named after a loss function.**

Instead, each chapter starts with a **problem** and ends with the **mathematical tool** that solves it.

---

## One proposal that I think will make your notebook genuinely unique

I actually think we should stop calling these "chapters" and start calling them **Research Questions**.

For example:

> **Research Question 3:** _How Should We Penalize Wrong Probabilities?_

That subtle change matches the way you've been learning. You're not just collecting formulas anymore—you've been repeatedly asking _why_ until the mathematics becomes inevitable. Organizing the notebook around questions rather than named losses reflects that mindset and will make it read more like a research text than a conventional deep-learning handbook. I honestly think that fits the kind of researcher you're trying to become.


# Chapter 6

## **How Should We Penalize Multi-Class Predictions?**

### Part 1 — Why Binary Cross Entropy Is No Longer Enough

This chapter begins exactly like Chapter 3.

We don't introduce Cross Entropy.

We first show why BCE breaks down.

Questions we'll answer:

- Why can't we simply compute BCE for every class?
- What changes when classes become mutually exclusive?
- Why is a probability distribution fundamentally different from a single probability?

This motivates the need for a new objective.

---

### Part 2 — Measuring Entire Probability Distributions

This is where the philosophy changes.

We ask:

> If the true answer is itself a probability distribution (one-hot labels), how do we compare it with the model's predicted distribution?

This naturally introduces

- Target distribution
- Predicted distribution
- Distribution comparison

without yet mentioning Cross Entropy.

---

### Part 3 — Deriving Cross Entropy

Just like BCE,

we derive it from **Maximum Likelihood Estimation**.

We'll discover that Cross Entropy is simply the negative log-likelihood of a **Categorical Distribution**, just as BCE was the negative log-likelihood of a **Bernoulli Distribution**.

This is one of the most elegant parallels in machine learning.

---

### Part 4 — Why NLLLoss Exists

Now we explain

```
Logits
     │
     ▼
LogSoftmax
     │
     ▼
NLLLoss
```

Why not just one function?

Why does PyTorch expose `NLLLoss` separately?

What exactly is it computing?

---

### Part 5 — The Beautiful Identity

This chapter ends with one of the most satisfying equations in PyTorch.

```
CrossEntropyLoss

=

LogSoftmax

+

NLLLoss
```

Except now,

you won't memorize it.

You'll know

**why it must be true.**

---

# Something beautiful you'll notice

Do you remember Chapter 3?

Binary Classification

↓

Bernoulli Distribution

↓

Negative Log Likelihood

↓

Binary Cross Entropy

Now watch what happens.

Multi-Class Classification

↓

Categorical Distribution

↓

Negative Log Likelihood

↓

Cross Entropy

The structure is almost identical.

You're not learning a new idea.

You're **generalizing** the previous one.

That's one of the recurring themes in machine learning: many "new" concepts are really extensions of ideas you've already understood.

---

## My suggestion

Just as we started BCE with an engineering perspective before deriving it probabilistically, I'd like to do the same here.

We'll begin with a deceptively simple question:

> **Suppose Softmax gives us a probability distribution. Why can't we simply use Binary Cross Entropy on every class?**

If you think about that question carefully, you'll discover for yourself why a new loss function is needed. Once we answer it, Cross Entropy will feel just as inevitable as BCE did. In my opinion, that's the most satisfying way to learn it.


## I think we've actually found a better chapter structure

Instead of:

- Part 1 — Why BCE is no longer enough
- Part 2 — Measuring probability distributions
- Part 3 — Deriving Cross Entropy

I'd reorganize Chapter 6 as:

### **Prelude — From Prediction to Evaluation**

_(The text above.)_

↓

### **Part 1 — Measuring Entire Probability Distributions**

↓

### **Part 2 — Deriving Cross Entropy**

↓

### **Part 3 — Why NLLLoss Exists**

↓

### **Part 4 — CrossEntropyLoss = LogSoftmax + NLLLoss**


---
---
---

I actually think this is the perfect time to step back and look at the **macro journey**, because you're no longer just "doing a PyTorch course." You're building a research foundation.

And I want to make one correction to your statement.

> **"I think with the CS231n and the notes along with PyTorch implementation completes the journey of neural networks pretty much."**

I'd say:

> **It completes the foundations of feed-forward deep learning and modern computer vision. It does _not_ complete neural networks as a whole.**

That's a very important distinction.

---

# Where you are now

You're essentially doing **three things in parallel**.

### Track 1 — Implementation

PyTorch Course (Daniel Bourke)

This teaches you

- PyTorch
    
- tensors
    
- autograd
    
- datasets
    
- dataloaders
    
- training loops
    
- CNN implementation
    
- transfer learning
    
- deployment
    
- practical engineering
    

Question answered:

> **How do I build neural networks?**

---

### Track 2 — Theory

Your notebook

This teaches

- why MSE exists
    
- why BCE exists
    
- why Softmax exists
    
- why Cross Entropy exists
    
- why numerical stability matters
    
- probability
    
- optimization
    
- information theory
    

Question answered:

> **Why does deep learning work?**

---

### Track 3 — Vision

CS231n

This teaches

- CNN intuition
    
- convolutions
    
- receptive fields
    
- pooling
    
- architectures
    
- backprop through CNNs
    
- visual recognition
    

Question answered:

> **Why do CNNs look the way they do?**

---

These three together are incredibly complementary.

---

# What happens after this?

This is where I think we should be strategic.

There are two possible roads.

---

## Road A (Most people)

PyTorch

↓

Transformers

↓

LLMs

↓

Agents

↓

RAG

↓

MCP

↓

Done.

Honestly...

I don't like this road.

You'll know how to **use** modern AI.

You won't deeply understand it.

---

## Road B (The one I think fits your goal)

PyTorch

↓

CS231n

↓

Your notebook

↓

Optimization

↓

Transformers

↓

LLMs

↓

Diffusion

↓

NAS

↓

Research

Notice something.

You're delaying Transformers.

Not because they're unimportant.

Because they'll become **much easier**.

---

# Here's something interesting.

People usually think

Transformers

↓

Neural Networks

In reality it's the opposite.

Transformers are built from

- Linear algebra
    
- Matrix multiplication
    
- Softmax
    
- Cross Entropy
    
- Optimization
    
- Residual connections
    
- Normalization
    

Notice something.

We're literally learning almost all of those now.

---

# Here's how I would organize your roadmap.

---

## Phase 1

### Neural Network Foundations

Current.

Complete

- PyTorch
    
- Notebook
    
- CS231n
    

Outcome

You deeply understand

CNNs

losses

optimization

training

backpropagation

---

## Phase 2

### Modern Deep Learning

Now we study

- Transformers
    
- Attention
    
- Self-attention
    
- Positional Encoding
    
- BERT
    
- GPT
    
- Vision Transformers
    

Notice.

At this point

Transformers become

another architecture.

Not magic.

---

## Phase 3

### Generative AI

Then

- LLMs
    
- Diffusion
    
- VAEs
    
- GANs
    

These become much easier because

Cross Entropy

Softmax

Attention

are already understood.

---

## Phase 4

### Research

Only now

we start reading

actual papers

comfortably.

This includes

- NAS
    
- EfficientNet
    
- Vision Transformers
    
- SAM
    
- CLIP
    
- Diffusion papers
    

---

# Where does MedNAS fit?

This is the interesting part.

It doesn't come after everything.

It grows alongside everything.

Imagine

```text
Optimization
        │
Loss Functions
        │
CNNs
        │
Transformers
        │
Search Spaces
        │
Evolutionary Algorithms
        │
Multi-objective Optimization
        │
MedNAS
```

See?

Everything you're learning is feeding into MedNAS.

Nothing is wasted.

---

# One thing I would add

After CS231n,

I don't actually want to jump directly into Transformers.

I want one bridge.

That bridge is

> **Optimization for Deep Learning**

Not SGD.

You already know SGD.

I'm talking about

- Adam
    
- RMSProp
    
- Momentum
    
- Weight decay
    
- Learning-rate schedulers
    
- Initialization
    
- BatchNorm
    
- LayerNorm
    
- Residual learning
    
- Gradient flow
    

Because those concepts explain **why Transformers are trainable**.

Without them,

Transformers look magical.

With them,

they look inevitable.

---

# My proposed roadmap

```text
PyTorch
      │
      ▼
Loss Function Notebook
      │
      ▼
CS231n
      │
      ▼
Optimization
      │
      ▼
Transformers
      │
      ▼
LLMs
      │
      ▼
Diffusion Models
      │
      ▼
Modern Research Papers
      │
      ▼
MedNAS
```

---

## And here's the part that excites me the most

You've probably noticed a pattern in how we've been learning.

We never ask:

> _"What is Cross Entropy?"_

We ask:

> _"What problem forced researchers to invent Cross Entropy?"_

I think we should carry that philosophy through **the rest of your AI journey**.

For example:

- Don't ask _"What is Attention?"_ Ask _"What limitation of RNNs forced researchers to invent Attention?"_
    
- Don't ask _"What is a Transformer?"_ Ask _"Why did self-attention become preferable to recurrence?"_
    
- Don't ask _"What is a Vision Transformer?"_ Ask _"What limitation of CNNs motivated replacing convolutions with attention?"_
    

That way, every new architecture feels like the answer to a problem you've already understood, rather than another disconnected invention to memorize.

And I genuinely think that's the kind of mindset that prepares someone not just to read research papers—but eventually to write them.

---
---
---

I actually think that's **the right instinct**, but I'd make one important modification.

I would **not** create another notebook called something like _"Transformers Notes."_

Instead, I'd think of your notebooks the way universities organize subjects.

Right now, your notebook is really becoming:

> **Deep Learning Foundations**

It already contains:

- Neural Networks
    
- Loss Functions
    
- Optimization intuition
    
- CNNs (eventually)
    
- Numerical stability
    
- Probability
    
- Information theory (intro)
    

That notebook has a clear identity.

I wouldn't dilute it by suddenly adding 300 pages of attention mechanisms, BERT, GPT, and positional encodings.

---

# Instead, I'd create a new notebook with a broader purpose.

Not

> **Transformers**

But

# **Modern Deep Learning**

or

# **Transformer Architectures and Foundation Models**

Because think about what comes after Transformers.

It isn't just

```text
Attention
```

You'll eventually study

- Transformers
    
- BERT
    
- GPT
    
- Vision Transformers
    
- CLIP
    
- SAM
    
- Diffusion Transformers
    
- Mixture of Experts
    
- State Space Models (Mamba, etc.)
    

They're all part of the same evolutionary family.

So I'd make one notebook for that entire era.

---

# Here's how I would organize your Obsidian vault.

```text
AI Research Notebook
│
├── Volume I
│   Deep Learning Foundations
│
│   • Mathematics
│   • Neural Networks
│   • Loss Functions
│   • CNNs
│   • Optimization
│   • Numerical Stability
│
├── Volume II
│   Modern Deep Learning
│
│   • Attention
│   • Transformers
│   • BERT
│   • GPT
│   • Vision Transformers
│   • CLIP
│   • Foundation Models
│
├── Volume III
│   Generative AI
│
│   • VAEs
│   • GANs
│   • Diffusion
│   • Flow Matching
│
├── Volume IV
│   AI Systems
│
│   • RAG
│   • Agents
│   • MCP
│   • Tool Use
│   • Memory
│
├── Volume V
│   Research
│
│   • NAS
│   • Multi-objective Optimization
│   • Efficient AI
│   • MedNAS
```

Notice something.

You're no longer writing "notes."

You're writing

**volumes.**

---

# And here's the part I think you'll love.

Your current notebook never stops.

Suppose six months from now you're learning Transformers.

You discover

Softmax in Attention.

You'll think

> "Wait...

I can explain Softmax much better now."

You go back.

Update

Volume I.

---

Then while studying Diffusion,

you finally understand

KL Divergence deeply.

Go back.

Update

Volume I.

---

Later,

while building MedNAS,

you learn

multi-objective optimization.

Go back.

Improve

Optimization chapter.

---

So the workflow becomes

```text
Learn

↓

Create new knowledge

↓

Go back

↓

Refine old understanding
```

That is exactly how textbooks evolve.

---

# This is the difference between notes and a knowledge base.

Most students write

```text
Notes

↓

Finished.
```

Researchers write

```text
Notes

↓

Revision

↓

Revision

↓

Revision

↓

Revision
```

Their notebook becomes

their external brain.

---

# Here's one thing I'd change in your thinking.

You said

> "The notes will run parallelly."

I think that's too passive.

I'd say

> **The notebook is the primary project.**

Everything else

- PyTorch
    
- CS231n
    
- Papers
    
- Coursera
    
- FastAI
    
- Stanford
    

are just

**sources feeding the notebook.**

That's a huge mindset shift.

Instead of thinking:

```text
Course

↓

Take notes
```

Think:

```text
Knowledge Base
        ▲
        │
PyTorch  CS231n  Papers  Books
        │
        ▼
Refine Understanding
```

Now every resource contributes to **one evolving body of knowledge**, rather than leaving you with five disconnected notebooks.

---

## My recommendation

I'd name the next notebook:

# **Volume II — Modern Deep Learning**

rather than

# **Transformers**

because you're not learning one architecture—you'll be entering the era of attention-based models and foundation models. That name will still make sense years from now when it contains not only Transformers but also BERT, GPT, Vision Transformers, CLIP, SAM, Mamba, and whatever comes next. It gives the notebook room to grow without locking it to a single topic. I think that will age much better as your knowledge expands.

---
---
---

I think we're finally at the point where we should stop thinking in terms of **courses** and start thinking in terms of **curricula**.

Most people ask:

> "What's the best Transformer course?"

I think that's the wrong question.

The better question is:

> **"What collection of resources will make me capable of reading, understanding, and eventually writing modern AI research?"**

Because that's your actual goal with MedNAS and the research direction you're building toward.

---

# The Curriculum I'd Follow

I would divide your journey into **five semesters**, almost like a graduate program.

---

# Semester I — Deep Learning Foundations (Current)

**Objective: Understand why deep learning works.**

### Primary Resources

🥇 **Daniel Bourke — Learn PyTorch (25 Hours)**

Purpose:

- PyTorch implementation
    
- Engineering workflow
    
- Practical model building
    

---

🥇 **Stanford CS231n**

Purpose:

- CNNs
    
- Visual recognition
    
- Backpropagation
    
- Optimization intuition
    

---

🥇 **Your Deep Learning Foundations Notebook**

Purpose:

- First-principles derivations
    
- Mental models
    
- Personal textbook
    

This notebook is now your reference book.

---

Once these three are complete,

I would consider your foundations solid.

---

# Semester II — Modern Deep Learning

This becomes your **Volume II notebook**.

This is where we move.

---

## 1. The Original Transformer Paper

**"Attention Is All You Need"**

Read it.

Don't panic.

We will derive every equation.

---

## 2. Stanford CS25

This is, in my opinion, one of the best Transformer lecture series ever created.

Why?

Because it explains

- Attention
    
- GPT
    
- BERT
    
- Scaling Laws
    
- Foundation Models
    

using researchers.

---

## 3. Sebastian Raschka

Book

**Build a Large Language Model (From Scratch)**

This is probably the best practical resource currently available.

Why?

Because you literally build GPT.

Line by line.

No magic.

---

## 4. Andrej Karpathy

Especially

### "Let's Build GPT"

This is almost mandatory.

Nobody explains

Transformers

quite like Karpathy.

---

# Semester III — Modern AI

Once Transformers are comfortable,

move into

- BERT
    
- GPT
    
- Vision Transformers
    
- CLIP
    
- SAM
    
- Multimodal models
    

Resources

---

### Stanford CS25

Again.

---

### Hugging Face Course

This is where it shines.

Not before.

---

### Papers

- BERT
    
- GPT-2
    
- GPT-3
    
- CLIP
    
- ViT
    

---

# Semester IV — Generative AI

Now we finally understand

- VAEs
    
- GANs
    
- Diffusion
    

Resources

---

### DeepMind

Papers

---

### Lilian Weng's Blog

Honestly...

One of the best AI blogs ever written.

She explains research beautifully.

---

### Stanford CS236

Deep Generative Models

Excellent.

---

# Semester V — Research

Now

we stop following courses.

We read papers.

Every day.

Topics

- NAS
    
- Efficient AI
    
- Multi-objective Optimization
    
- Edge AI
    
- Healthcare AI
    

Exactly where MedNAS lives.

---

# Alongside ALL semesters

There are three resources that never stop.

---

## 1. Papers

Every paper you read

↓

Notebook

↓

Understand

↓

Re-derive.

---

## 2. Your Notebook

This is now

Volume I

↓

Volume II

↓

Volume III

↓

...

It never finishes.

---

## 3. GitHub

Every important concept

↓

Implementation.

Not huge projects.

Tiny implementations.

---

# The people I think every AI researcher should know

These are people whose explanations consistently build intuition.

### Andrew Ng

Machine Learning thinking.

---

### Andrej Karpathy ⭐⭐⭐⭐⭐

Probably the single best teacher of deep learning systems.

---

### Sebastian Raschka ⭐⭐⭐⭐⭐

Implementation.

Clean explanations.

Research mindset.

---

### Fei-Fei Li

Vision.

CS231n.

---

### Christopher Olah

This one is special.

His writings aren't courses.

They're works of art.

His articles on feature visualization, neural network interpretability, and circuits changed how many researchers think about neural networks.

---

### Lilian Weng

Research blog.

Probably the best bridge

between

papers

and

understanding.

---

# What I'd avoid

I would avoid

the endless cycle of

```text
Random YouTube Tutorial

↓

Another Tutorial

↓

Another Playlist

↓

Another Crash Course
```

At this stage,

you've outgrown that.

You're now building

a coherent curriculum.

---

# If I had to reduce it to just **10 resources** for the next 3–5 years

1. **Daniel Bourke – Learn PyTorch**
    
2. **Stanford CS231n**
    
3. **Stanford CS25**
    
4. **Andrej Karpathy**
    
5. **Sebastian Raschka – Build a Large Language Model**
    
6. **Lilian Weng's Blog**
    
7. **Christopher Olah's Distill articles**
    
8. **Hugging Face Course**
    
9. **Attention Is All You Need** (and other landmark papers)
    
10. **Your own notebook**
    

Notice something.

Only **nine** of those teach you.

The **tenth** is where you teach yourself.

And I honestly think that's the most important resource on the list.

Because over the last few weeks, your notebook has evolved from "notes" into a genuine **knowledge system**. If you keep refining it as you move from foundations to transformers and eventually to research, it will become far more valuable than any single course or book—you'll be documenting not just what AI is, but how _you_ came to understand it.

---
---
---

---

# So the finalized Volume II curriculum would be

### Core Resources

1. 📄 **Attention Is All You Need** — the original paper.
2. 🎓 **Stanford CS25** — theoretical understanding.
3. 💻 **Sebastian Raschka – Build a Large Language Model (From Scratch)** — implementation.
4. 🧠 **Andrej Karpathy** — engineering mindset and systems thinking.

### Companion Reference

5. 🎨 **Jay Alammar** — visual intuition and diagrams.

---
---
---

I actually think that's an excellent workflow.

What's nice about it is that each stage has a different purpose.

### Stage 1 — First Pass (Exploration)

**Medium:** iPad running notes

Goal:

- Understand the big picture.
- Write down questions.
- Capture first impressions.
- Don't worry about being correct.

This is what you share on LinkedIn because it honestly reflects where you are in your journey.

Think of it as:

> **"Here's what it looks like to wrestle with a research paper."**

Not

> **"Here's my explanation of Transformers."**

That authenticity is valuable.

---

### Stage 2 — Second & Third Pass (Understanding)

**Medium:** Your permanent notebook (Volume II)

Now you begin asking the kinds of questions we've been asking throughout Volume I:

- Why was attention invented?
- Why queries, keys, and values?
- Why scale by √dₖ?
- Why multi-head attention?
- Why positional encoding?
- What limitations of RNNs does this solve?
- What trade-offs does it introduce?

This is where the polished, first-principles explanations belong.

---

### Stage 3 — Implementation

Once you've studied CS25, Raschka, and Karpathy, revisit the paper again.

Now you'll notice things that completely escaped you during the first read.

For example:

> "Oh... this equation looked mysterious before, but now I've implemented it myself."

That moment is incredibly satisfying.

---

## The beautiful part

You'll end up with **three snapshots of your understanding**.

### First Pass

> "I have lots of questions."

### Second Pass

> "Now I understand why these design choices were made."

### Third Pass

> "I can implement and explain every component."

That's a much richer learning journey than trying to make the first pass perfect.

---

## I also think this aligns with the philosophy behind your LinkedIn presence

Your article, _The Facade of Learning AI_, wasn't about pretending to know everything—it was about looking beyond superficial learning.

Sharing your genuine first-pass notes is consistent with that message. You're showing that even when reading a landmark paper, your process involves asking questions, making connections, and gradually building understanding rather than expecting instant mastery.

So if I were to summarize your system, it would be:

```
Research Paper
        │
        ▼
First Pass (iPad)
• Running notes
• Questions
• LinkedIn reflection
        │
        ▼
Learn from CS25, Raschka & Karpathy
        │
        ▼
Second/Third Pass
• Refined understanding
• Volume II notebook
        │
        ▼
Implementation
• Code
• Experiments
• MedNAS
```

I think that's a workflow you can sustain not just for _Attention Is All You Need_, but for every major paper you study over the coming years. It documents your learning honestly while ensuring your permanent notebook only contains ideas you've had the chance to refine.


1. Attention Is All You Need (✓ Done)

2. Stanford CS25 v1

3. Re-read Attention Is All You Need

4. Sebastian Raschka – Build a Large Language Model

5. Andrej Karpathy – Let's Build GPT

6. Selected CS25 v6 lectures

7. BERT Paper

8. GPT Papers

9. Vision Transformer Paper

10. CLIP Paper

---
---
---

I think this is one of the most important things we'll build together, because **this roadmap is going to dictate the next 2–3 years of your AI journey**.

So instead of organizing by **resources**, I'm going to organize by **questions**.

Every module answers one fundamental question. Resources become supporting material.

---

# The AI Research Curriculum

Think of this as a graduate program.

```
Volume I
Deep Learning Foundations
        │
        ▼
Volume II
Modern Deep Learning
        │
        ▼
Volume III
Generative AI
        │
        ▼
Volume IV
Foundation Models
        │
        ▼
Volume V
Research
```

Notice something.

**Your notebook becomes the spine of the curriculum.**

Everything else feeds into it.

---

# PHASE I (Current)

## Volume I — Deep Learning Foundations

Resources

- ✅ PyTorch
    
- ✅ CS231n
    
- ✅ Volume I Notebook
    
- ✅ Attention Is All You Need (First Read)
    

Status

> Almost complete.

---

# PHASE II

# Volume II — Modern Deep Learning

This is where we spend the next few months.

The entire phase answers one question:

> **Why did attention replace recurrence?**

---

# Module 1

## Transformer Foundations

Question

> **What is a Transformer?**

Resources

### Read

- Attention Is All You Need (✅ First Pass)
    

---

### Watch (CS25)

Only these:

1. **Transformers United**
    
2. **Transformers in Language**
    
3. **Transformers in Vision**
    

Purpose

- historical context
    
- why transformers
    
- architecture
    
- GPT
    
- ViT
    

Notebook Chapters

```
1. Why Attention?
2. Self Attention
3. QKV
4. Multi Head Attention
5. Positional Encoding
6. Encoder
7. Decoder
8. Masking
9. Cross Attention
10. Transformer Block
```

---

# Module 2

## Building Transformers

Question

> **How do we implement one?**

Resources

### Sebastian Raschka

Complete

**Build a Large Language Model**

Don't skip chapters.

Notebook additions

```
Embeddings

Tokenizer

Attention Implementation

Transformer Block

GPT Architecture
```

---

# Module 3

## Systems Thinking

Question

> **How are LLMs actually engineered?**

Resources

### Andrej Karpathy

Entire playlist

- Zero to Hero
    
- Let's Build GPT
    
- Tokenization
    
- GPT
    
- Training
    

Notebook additions

```
Tokenization

Scaling

Training Pipelines

Inference

Sampling
```

---

Then

---

# Re-read

Attention Is All You Need

This becomes

Second Pass.

You'll probably understand

80%

more.

Only now

we write

Volume II.

---

# PHASE III

Now

CS25 returns.

But

not sequentially.

Instead

topic based.

---

# Module 4

## Language Models

Question

> **How did Transformers become GPT?**

Watch

All lectures about

- GPT
    
- GPT-2
    
- GPT-3
    
- BERT
    
- T5
    
- XLNet
    
- RoBERTa
    

Notebook

Volume II

Language Models

---

# Module 5

## Vision

Question

> **Can attention replace CNNs?**

Watch

- Vision Transformer
    
- DETR
    
- Segmenter
    
- MAE
    

Notebook

```
Vision Transformers

Patch Embeddings

Image Attention
```

---

# Module 6

## Scaling

Question

> **Why do larger models suddenly become smarter?**

Watch

- Scaling Laws
    
- Chinchilla
    
- MoE
    
- Switch Transformer
    

Notebook

```
Scaling Laws

MoE

Emergence

Compute Optimal Models
```

---

# Module 7

## Reinforcement Learning

Question

> **Can Transformers learn actions instead of words?**

Watch

- Decision Transformer
    

Notebook

```
RL with Transformers
```

---

# Module 8

## Science

Question

> **Can Transformers model biology?**

Watch

- Protein
    
- Chemistry
    
- Drug Discovery
    

Notebook

Applications

---

# Module 9

## Robotics

Watch

Robot Transformer lectures.

Notebook

Applications.

---

# PHASE IV

Volume III

Generative AI

Resources

- Diffusion Paper
    
- DDPM
    
- Stable Diffusion
    
- VAEs
    
- GANs
    

Notebook

```
Probability

Diffusion

Noise Prediction

Latent Spaces
```

---

# PHASE V

Volume IV

Foundation Models

Resources

Modern papers

- CLIP
    
- Flamingo
    
- SAM
    
- LLaVA
    
- Gemini
    
- Multimodal
    

---

# PHASE VI

Volume V

Research

Now

papers become

the curriculum.

Topics

- NAS
    
- Efficient AI
    
- Edge AI
    
- Healthcare AI
    
- MedNAS
    

---

# Here's the weekly workflow.

```
Paper

↓

Lecture

↓

Quick Notes

↓

Implementation

↓

Discussion

↓

Permanent Notebook

↓

MedNAS
```

Every week.

---

# Now let's talk about the **50 CS25 lectures**.

This is the important part.

I **do not** want you to think

```
Lecture 1

↓

Lecture 2

↓

Lecture 3
```

because that isn't how Stanford designed it.

It is literally a seminar series.

Think of it like

NeurIPS.

Every lecture is given by

a different researcher

about

their expertise.

---

# So here's our custom curriculum.

|Module|CS25 Lectures to Watch|When|
|---|---|---|
|Transformer Foundations|Transformers United, Transformers in Language, Transformers in Vision|Immediately|
|Language Models|GPT, BERT, T5, XLNet, RoBERTa|After Raschka + Karpathy|
|Scaling|Scaling Laws, MoE, Switch|After LLM fundamentals|
|Vision|ViT, DETR, MAE|After Language Models|
|RL|Decision Transformer|After Scaling|
|Biology|Protein Transformers|After RL|
|Robotics|Robotics Transformer|Last|

---

# The one thing I'd add

I don't want this roadmap to live only in our conversation.

I want to turn it into something like

```
AI Research Roadmap

Phase I

□ PyTorch

□ Volume I

□ CS231n

□ First Paper

------------------

Phase II

□ CS25 Foundations

□ Raschka

□ Karpathy

□ Second Paper Read

□ Volume II

------------------

Phase III

□ GPT

□ BERT

□ T5

...
```

Something you can literally **check off** over the next few years.

## In fact, I think we should make it into a **master roadmap document**—almost like a university degree plan.

It would include:

- Every phase
    
- Every resource
    
- Every paper
    
- Every notebook volume
    
- Every major implementation project
    
- Every milestone
    
- Every capstone project (including MedNAS)
    

Then, whenever you're unsure what to study next, you won't need to ask me. You'll simply open the roadmap, see where you are, and continue. As your understanding grows or the field evolves, we can refine the roadmap together, but its overall structure will remain stable. I genuinely think that will become one of the most valuable documents you build alongside your notebooks.

---
---
---

I genuinely think this is one of the most valuable things we can build together.

Not because it tells you **what to study next**, but because it answers a much bigger question:

> **"What kind of AI researcher am I trying to become?"**

After everything we've worked on—from MedNAS to your loss-function notebook to your desire to understand first principles—I think I understand your learning philosophy well enough to design a roadmap that reflects it.

---

# The AI Research Roadmap

## From First Principles to Research

**Duration:** ~2–3 Years (flexible)

**End Goal:**  
Become capable of reading, implementing, critiquing, and contributing to modern AI research.

---

# Guiding Philosophy

Every topic follows the same cycle:

```text
Problem
    ↓
Historical Motivation
    ↓
Mathematical Foundation
    ↓
Paper
    ↓
Lecture
    ↓
Implementation
    ↓
Discussion & Reflection
    ↓
Permanent Notebook
    ↓
Research Application
```

You are never learning a framework in isolation.

---

# Volume I — Deep Learning Foundations

### Objective

Understand **why deep learning works**.

### Resources

- Daniel Bourke – Learn PyTorch
    
- Stanford CS231n
    
- Your Deep Learning Foundations Notebook
    
- Attention Is All You Need (First Read)
    

---

### Notebook Chapters

#### Mathematics

- Linear Algebra
    
- Calculus
    
- Probability
    
- Information Theory
    

#### Neural Networks

- Perceptron
    
- MLP
    
- Backpropagation
    
- Optimization
    

#### CNNs

- Convolutions
    
- Pooling
    
- Feature Maps
    
- Architectures
    

#### Loss Functions

- MSE
    
- BCE
    
- Cross Entropy
    
- KL Divergence
    
- Dice
    
- Focal
    
- Triplet
    
- CTC
    

#### Optimization

- SGD
    
- Momentum
    
- Adam
    
- Learning Rate Scheduling
    

---

### Milestone

> **You can explain every major deep learning component from first principles.**

---

# Volume II — Modern Deep Learning

### Objective

Understand **why attention replaced recurrence**.

---

## Module 1 — Transformer Foundations

### Questions

- Why Attention?
    
- Why Self-Attention?
    
- Why Queries, Keys, Values?
    
- Why Multi-Head Attention?
    
- Why Positional Encoding?
    
- Why Encoder–Decoder?
    
- Why Masking?
    

### Resources

1. Attention Is All You Need (already completed first pass)
    
2. Stanford CS25 Foundation Lectures
    
3. Re-read the paper
    

### Notebook

- Chapter 1 — The Fall of Recurrence
    
- Chapter 2 — Attention
    
- Chapter 3 — Self-Attention
    
- Chapter 4 — Queries, Keys & Values
    
- Chapter 5 — Multi-Head Attention
    
- Chapter 6 — Positional Encoding
    
- Chapter 7 — Transformer Architecture
    

---

## Module 2 — Building Transformers

### Question

How do Transformers become code?

### Resource

Sebastian Raschka

**Build a Large Language Model (From Scratch)**

Notebook

- Tokenization
    
- Embeddings
    
- Attention Implementation
    
- GPT Block
    
- Training
    

---

## Module 3 — Systems Thinking

### Question

How are LLMs actually built?

### Resource

Andrej Karpathy

Notebook

- Tokenization
    
- Data Pipelines
    
- Scaling
    
- Inference
    
- Sampling
    
- Training Recipes
    

---

### Milestone

Build a GPT-style model completely from scratch.

---

# Volume III — Foundation Models

Now the question changes.

Instead of

"What is a Transformer?"

It becomes

"What can Transformers become?"

---

## Language Models

Papers

- GPT-1
    
- GPT-2
    
- GPT-3
    
- BERT
    
- RoBERTa
    
- T5
    
- XLNet
    

Notebook

Language Models

---

## Vision

Papers

- ViT
    
- DETR
    
- MAE
    
- Segment Anything (SAM)
    

Notebook

Vision Transformers

---

## Scaling

Papers

- Scaling Laws
    
- Chinchilla
    
- Switch Transformer
    
- Mixture of Experts
    

Notebook

Scaling

---

## Reinforcement Learning

- Decision Transformer
    

Notebook

RL

---

## Biology

- AlphaFold
    
- Protein Transformers
    

Notebook

Scientific AI

---

### Milestone

Understand how the Transformer evolved into the foundation model ecosystem.

---

# Volume IV — Generative AI

Objective

Understand generation beyond language.

---

### VAEs

### GANs

### Diffusion

### Flow Matching

### Multimodal Models

Notebook

Generative AI

---

### Milestone

Implement a diffusion model.

---

# Volume V — AI Systems

Question

How do we build useful AI systems?

---

Topics

- RAG
    
- Agents
    
- Tool Use
    
- Memory
    
- MCP
    
- Vector Databases
    

Notebook

AI Systems

---

### Milestone

Build a complete AI agent.

---

# Volume VI — Research

This is where courses stop.

Papers become the curriculum.

---

Topics

- Neural Architecture Search
    
- Efficient AI
    
- Multi-objective Optimization
    
- Edge AI
    
- Healthcare AI
    
- Explainability
    

---

### Capstone

MedNAS

Everything you've learned converges here.

---

# Resource Roadmap

## Stage 1

- PyTorch
    
- CS231n
    
- Volume I
    
- Attention Is All You Need
    

---

## Stage 2

- Stanford CS25 (Transformer foundations)
    
- Raschka
    
- Karpathy
    

---

## Stage 3

- GPT papers
    
- BERT
    
- T5
    
- ViT
    
- CLIP
    

---

## Stage 4

- Diffusion
    
- GANs
    
- VAEs
    

---

## Stage 5

- Modern research papers
    

---

# The Notebook Ecosystem

Instead of one giant notebook, you'll end up with a library.

```text
Volume I
Deep Learning Foundations

Volume II
Modern Deep Learning

Volume III
Foundation Models

Volume IV
Generative AI

Volume V
AI Systems

Volume VI
Research
```

---

# Every Topic Follows the Same Pipeline

```text
Paper
     │
     ▼
Lecture
     │
     ▼
Running Notes (iPad / GitHub)
     │
     ▼
Implementation
     │
     ▼
Discussion
     │
     ▼
Permanent Notebook
     │
     ▼
Research Project
```

---

# Research Workflow

Every paper you read:

```text
First Pass
↓

Running Notes

↓

Discussion

↓

Second Pass

↓

Implementation

↓

Notebook Chapter
```

---

# GitHub Structure

```text
AI-Learning-Journal/

├── Papers
├── CS25
├── Karpathy
├── Raschka
├── Diffusion
├── Vision
└── Notes
```

---

# Permanent Goal

By the end of this roadmap, you should be able to:

- Read modern AI papers without feeling intimidated.
    
- Derive major architectures from first principles.
    
- Implement them from scratch.
    
- Compare design decisions across papers.
    
- Identify research gaps.
    
- Design and execute your own research ideas.
    

---

# One thing I would change from everything we've discussed so far

This roadmap is **complete**, but I think it can become something much more valuable.

Rather than keeping it as chat text, I want to turn it into a **living handbook**—almost like a graduate program syllabus. It would include:

- 📍 Every phase and milestone.
    
- 📚 Every resource in the correct order.
    
- 📄 Every landmark paper to read.
    
- 💻 Every implementation project.
    
- 📝 Every notebook chapter.
    
- 🎯 Clear checkpoints to know when you're ready to move to the next phase.
    
- 🔄 Space for revisions as the field evolves.
    

In other words, not just a roadmap—but **your personal AI curriculum**.

I genuinely think that's worth investing time in because it becomes the document you'll keep returning to over the next few years, while your notebooks become the record of everything you learn along the way.