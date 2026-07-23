
# Recurrent Neural Networks (RNN)

> _"A Recurrent Neural Network extends a feed-forward neural network by introducing a hidden state that is carried across time, enabling the network to process sequential data while maintaining contextual information from previous inputs."_

---

# Why RNNs?

Traditional neural networks assume that inputs are **independent**.

Examples:

- Image Classification
- Tabular Data
- Regression

However, many real-world problems involve **sequences**, where previous information influences future predictions.

Examples:

- Language Modeling
- Machine Translation
- Speech Recognition
- Time Series Forecasting
- Image Captioning

The order of the data matters.

---

# Motivation

Suppose we want to predict the next word in

```
The cat sat on the _____
```

The prediction depends on

```
The
↓

cat
↓

sat
↓

on
```

A feed-forward network cannot naturally remember previous words.

An RNN introduces memory through a **hidden state**.

---

# Core Idea

Instead of treating every input independently,

the network keeps a running summary of everything it has seen so far.

```
x₁ → h₁

↓

x₂ → h₂

↓

x₃ → h₃

↓

...

↓

xₜ → hₜ
```

where

- xₜ = current input
- hₜ = hidden state

---

# Hidden State

The hidden state acts as the network's memory.

At every time step,

the new hidden state depends on

- the current input
- the previous hidden state

Conceptually,

```
Current Memory

=

Previous Memory

+

Current Information
```

Mathematically,

```
hₜ = f(hₜ₋₁, xₜ)
```

A common implementation is

```
hₜ = tanh(Wₕₕ hₜ₋₁ + Wₓₕ xₜ)
```

---

# Important Property

The **same parameters** are reused at every time step.

Unlike feed-forward networks,

an RNN does **not** learn separate weights for every position.

One set of parameters processes the entire sequence.

Benefits:

- fewer parameters
- variable-length sequences
- generalization across sequence lengths

---

# Sequence Processing

RNNs support several input-output configurations.

---

## One → One

```
Image

↓

Label
```

Example:

Image Classification

---

## One → Many

```
Image

↓

Caption
```

Example:

Image Captioning

---

## Many → One

```
Sentence

↓

Sentiment
```

Example:

Sentiment Analysis

---

## Many → Many (Aligned)

```
Video Frames

↓

Action Labels
```

Example:

Frame-wise Video Classification

---

## Many → Many (Encoder–Decoder)

```
English

↓

French
```

Example:

Machine Translation

The encoder summarizes the input sequence into a context vector.

The decoder generates the output sequence.

---

# Unrolling the Network

Although drawn as one recurrent block,

an RNN is actually **unrolled through time**.

```
      h₀

↓

x₁ → h₁

↓

x₂ → h₂

↓

x₃ → h₃

↓

...
```

Every time step uses the same network.

Only the hidden state changes.

---

# Training

Training uses

**Backpropagation Through Time (BPTT).**

Steps:

1. Forward pass through the entire sequence.
2. Compute loss.
3. Propagate gradients backward through every time step.

---

# Truncated BPTT

Long sequences become computationally expensive.

Instead of backpropagating through the entire sequence,

training is often performed on smaller chunks.

Benefits:

- lower memory usage
- faster training
- practical for long sequences

---

# Language Modeling

One of the earliest successful applications of RNNs.

Training:

```
Input

h

↓

Target

e

-------

Input

e

↓

Target

l

-------

Input

l

↓

Target

l

-------

Input

l

↓

Target

o
```

The network learns to predict the next token.

During inference,

the predicted token becomes the next input.

This process is called **autoregressive generation**.

---

# Strengths

- Handles variable-length sequences.
- Shares parameters across time.
- Captures temporal dependencies.
- Suitable for language, speech, and time-series tasks.

---

# Limitations

## Sequential Computation

Every hidden state depends on the previous one.

```
h₁

↓

h₂

↓

h₃

↓

h₄
```

This prevents parallel computation.

---

## Long-Term Dependencies

Information must travel through many hidden states.

```
Word 1

↓

Word 2

↓

...

↓

Word 100
```

Important information may gradually disappear.

---

## Vanishing / Exploding Gradients

During BPTT,

gradients are repeatedly multiplied.

Very small gradients

↓

Vanishing Gradient

Very large gradients

↓

Exploding Gradient

Both make optimization difficult.

---

# Evolution

The limitations of vanilla RNNs led to more advanced architectures.

```
Vanilla RNN

↓

LSTM

↓

GRU

↓

Attention

↓

Transformer
```

---

# Why Transformers Replaced RNNs

|RNN|Transformer|
|---|---|
|Sequential computation|Fully parallel computation|
|Hidden state carries information|Self-attention connects all tokens directly|
|Long path between distant words|Constant path length|
|Difficult to learn long-range dependencies|Direct interaction between all positions|
|Limited GPU utilization|Excellent parallelization|

---

# Key Takeaways

- RNNs introduce memory through a hidden state.
- The hidden state is updated recursively at every time step.
- Parameters are shared across the entire sequence.
- Training uses Backpropagation Through Time.
- Sequential computation limits parallelization.
- Difficulty learning long-range dependencies motivated attention mechanisms and ultimately the Transformer.