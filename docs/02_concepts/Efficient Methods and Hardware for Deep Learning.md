
# CS231n Lecture 15: Efficient Methods and Hardware for Deep Learning

## Core Idea

Deep learning models keep getting better because we keep increasing:

- Data
- Model size
- Computation

Examples:

- AlexNet → ResNet
- Deep Speech 1 → Deep Speech 2

But bigger models create three major problems:

### 1. Model Size

Large models are difficult to deploy on phones and edge devices.

Example:

- AlexNet ≈ 60 million parameters
- VGG ≈ 130 million parameters

Large models consume huge amounts of memory.

---

### 2. Speed

Training large models takes days or weeks.

Researchers spend more time waiting than experimenting.

Example:

- ResNet-18 ≈ 2.5 days
- ResNet-152 ≈ 1.5 weeks

Training time becomes a bottleneck.

---

### 3. Energy

Energy is often the hidden cost.

Examples:

- Mobile phones drain battery
- Data centers spend huge electricity costs
- AlphaGo reportedly used massive compute resources

The surprising fact:

> Memory access consumes more energy than computation.

A DRAM access can cost hundreds of times more energy than an arithmetic operation.

---

# Algorithm-Hardware Co-Design

Traditional approach:

```
Algorithm Team      ↓Hardware Team
```

Hardware treats the algorithm as a black box.

Song Han argues:

```
Algorithm ↔ Hardware
```

Design them together.

This is called:

**Algorithm-Hardware Co-Design**.

---

# Hardware Basics

The lecture introduces the hardware family tree.

### CPU

Characteristics:

- Flexible
- Low latency
- General purpose

Good for:

- Sequential tasks
- Diverse workloads

---

### GPU

Characteristics:

- Massive parallelism
- High throughput

Good for:

- Matrix multiplication
- Deep learning training

---

### FPGA

Characteristics:

- Programmable hardware

Middle ground between CPU and ASIC.

---

### ASIC

Characteristics:

- Custom-built hardware
- Fastest
- Most efficient
- Least flexible

Examples:

- TPU
- Dedicated AI chips

---

# Number Representations

Different numerical formats trade accuracy for efficiency.

### FP32

Standard deep learning format.

Pros:

- High accuracy
- Large range

Cons:

- More memory
- More energy

---

### FP16

Half precision.

Pros:

- Faster
- Smaller
- Less memory

Cons:

- Reduced precision

---

### INT8

Very efficient.

Pros:

- Tiny memory footprint
- Fast inference

Cons:

- Accuracy may drop

---

## Key Lesson

Lower precision:

```
Less Memory↓Less Energy↓Higher Speed
```

This principle appears repeatedly throughout the lecture.

---

# Part 1: Algorithms for Efficient Inference

Inference = using a trained model.

The lecture covers six major techniques.

---

# 1. Pruning

Observation:

Many weights contribute almost nothing.

Idea:

Remove tiny weights.

Process:

```
Train↓Prune↓Retrain
```

Benefits:

- Smaller model
- Less memory
- Faster inference

The famous result:

AlexNet can lose roughly 90% of its connections while maintaining accuracy after retraining.

---

### Why Pruning Works

Deep networks are heavily over-parameterized.

Many connections are redundant.

Pruning reveals the sparse network hidden inside the dense network.

---

### Rule of Thumb

The lecture repeatedly implies:

```
0 × A = 0
```

If a weight is zero:

- Don't store it
- Don't multiply it

Huge savings result.

---

# 2. Weight Sharing

After pruning:

Many remaining weights are similar.

Instead of storing:

```
0.480.490.470.50
```

Store:

```
Cluster Center = 0.48
```

and keep only an index.

Benefits:

- Fewer unique values
- Smaller storage

Think:

```
Many roads↓One shared address
```

---

# 3. Quantization

Idea:

Reduce precision.

Instead of:

```
32-bit weights
```

use:

```
16-bit8-bit4-bit
```

Benefits:

- Less memory
- Less bandwidth
- Faster arithmetic

Tradeoff:

- Potential accuracy loss

---

# 4. Low-Rank Approximation

Observation:

Weight matrices often contain redundancy.

Replace:

```
Large Matrix
```

with

```
Small Matrix A×Small Matrix B
```

Benefits:

- Fewer parameters
- Fewer computations

Conceptually similar to compression.

---

# 5. Binary and Ternary Networks

Extreme quantization.

### Binary

Weights:

```
-1+1
```

only.

---

### Ternary

Weights:

```
-10+1
```

only.

Benefits:

- Tiny models
- Very fast operations

Tradeoff:

- Harder to maintain accuracy

The lecture discusses Trained Ternary Quantization (TTQ) as a successful approach.

---

# 6. Winograd Convolution

Convolution is expensive.

Instead of performing standard convolution directly:

Transform the computation into a more efficient form.

Key result:

For a common 3×3 convolution:

```
Direct:     36C operationsWinograd:   16C operations
```

About:

```
2.25× fewer multiplications
```

Same output.

Less math.

More speed.

---

# Hardware for Efficient Inference

The lecture then moves into specialized hardware.

Examples:

- TPU
- Eyeriss
- DaDianNao
- EIE

---

# Most Important Hardware Insight

Memory dominates energy.

Approximate intuition:

```
Compute = cheapMemory = expensive
```

Especially:

```
DRAM access
```

is extremely costly.

Therefore:

The best accelerator is often the one that minimizes memory movement rather than arithmetic.

---

# Deep Compression

Song Han's famous pipeline:

```
Train↓Prune↓Quantize↓Huffman Coding
```

Goal:

Fit large networks into on-chip memory.

Benefits:

- Smaller
- Faster
- More energy efficient

---

# EIE (Efficient Inference Engine)

A custom ASIC built specifically for compressed neural networks.

Main idea:

Instead of accelerating dense networks,

accelerate compressed networks directly.

Advantages:

- Exploit sparsity
- Exploit weight sharing
- Skip zero activations
- Avoid expensive DRAM accesses

Reported improvements were dramatic over CPUs and GPUs.

---

# Part 3: Efficient Training

Now the focus shifts from inference to training.

---

# 1. Parallelization

Deep learning naturally contains huge amounts of parallel work.

---

### Data Parallelism

Idea:

Run different training examples on different GPUs.

```
GPU1 → Batch AGPU2 → Batch BGPU3 → Batch C
```

Then combine gradients.

Most common modern approach.

---

### Model Parallelism

Idea:

Split the model itself.

Example:

```
Half network → GPU1Half network → GPU2
```

Useful when model is too large for one GPU.

---

### Hyperparameter Parallelism

Run many experiments simultaneously.

Example:

```
LR=0.001LR=0.0001LR=0.01
```

all at once.

---

### Key Lesson

DNNs contain enormous parallelism.

Training scale comes from exploiting it.

---

# 2. Mixed Precision Training

One of the most influential ideas from the lecture.

Use:

```
FP16
```

for most computations.

Keep:

```
FP32
```

master weights.

Why?

FP16:

- Faster
- Smaller

FP32:

- Stable updates

Best of both worlds.

---

# 3. Model Distillation

Idea:

Train a small model using a large model.

Terminology:

```
Teacher Model↓Student Model
```

The student learns from teacher predictions.

Result:

- Smaller model
- Similar performance

Think:

```
Expert teaches apprentice
```

---

# 4. DSD Training

Dense → Sparse → Dense

Process:

```
Train Dense↓Prune to Sparse↓Retrain Dense
```

Why?

Pruning forces the network to escape poor local minima.

Result:

Often better accuracy than the original model.

Very interesting result:

Compression can sometimes improve performance.

---

# Hardware for Efficient Training

The lecture ends with hardware trends.

---

### CPUs

Adding AI-specific instructions.

Goal:

Make CPUs better for deep learning workloads.

---

### GPUs

Dominant training hardware.

Reasons:

- Massive parallelism
- High memory bandwidth

Examples:

- Pascal
- Volta

---

### Tensor Cores

Major Volta innovation.

Specialized matrix multiplication hardware.

Benefits:

- Huge throughput increase
- Designed for mixed precision training

This idea eventually became fundamental to modern AI hardware.

---

# The 10 Things to Remember Six Months From Now

1. Bigger models create size, speed, and energy problems.
2. Memory access is often more expensive than computation.
3. Efficient AI requires algorithm-hardware co-design.
4. Pruning removes unnecessary weights.
5. Quantization reduces numerical precision for speed and memory savings.
6. Binary/Ternary networks push quantization to the extreme.
7. Winograd reduces convolution computation.
8. Parallelization is the key to scaling training.
9. Mixed precision (FP16 + FP32) is now standard practice.
10. Future AI progress depends not only on better models but also on better hardware and efficiency techniques.

---

### MedNAS Connection (important)

This lecture is almost a blueprint for where MedNAS can evolve.

Your future objective function shouldn't be:

```
Maximize Accuracy
```

but:

```
Maximize AccuracyMinimize ParametersMinimize LatencyMinimize EnergyMinimize Memory
```

That is exactly the philosophy behind efficient deep learning and deployment-aware NAS that Song Han is advocating throughout this lecture.