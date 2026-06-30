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