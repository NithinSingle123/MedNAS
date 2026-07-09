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