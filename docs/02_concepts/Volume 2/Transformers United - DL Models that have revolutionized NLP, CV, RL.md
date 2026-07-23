# Transformers

- Before we used to have RNNs and LSTMs, before the paper "Attention is all you need" (Vaswani et. al)

### Attention 

#### Soft Attention

Soft attention: learn attention weight in [0, 1] over image patches

Cons: Expensive computation

#### Hard Attention

Hard Attention: learn attention weights in (0, 1) over image patches

Cons: Non-differentiability

#### Global Attention models

- Calculate attention over all of the sequence
- Similar to soft attention mechanism

#### Local attention models

- Calculate attention over a small window of sequence
- Combines local hard attention with global soft-attention

#### Self Attention

- Think of it as a search retrieval problem:
	- Given a query q find the set of keys k most similar to q and return the corresponding values v
	- q, k and v are all drawn from the same source. e.g. we can have q=k=v=x (where x is the output of the previous layer)
		- In transformers they are obtained by applying different linear transformations to x
	- Attention =  


#### Other necessary ingredients for the recipe

##### Positional representations/embeddings

- Impart the notion of "ordering", since self-attention is an unordered function of its inputs.

##### Nonlinearities

- Implemented as a simple feed-forward network. As with any other deep learning model, this allows for more complex mappings between input and output

##### Masking

- In order to parallelize operations while not looking at the future
- Keeps information about the future from "leaking" to the past
- Used in the decoder


#### Encoder - Decoder architecture

- Similar to language modelling tasks using LSTMs, transformers follow an encoder-decoder architecture
- The transformer in Vaswani, et.al, stacks 6 encoder blocks followed by 6 decoder blocks
- Each encoder blocks has:
	- A self-attention layer followed by a feedforward layer.
	- Each of these is also followed by a layer norm
	- There are residual connections between encoder blocks

- Each decoder is similar to the encoder, except that it has a 3rd layer that performs multi head attention on the output from the encoders. Masking is used here to prevent positions looking ahead during self-attention.

## Advantages

- constant path length between any two positions in the sequence (hence better to model context in long sequences)

- Transformers lend themselves better to parallelization (no sequential computation within a layer)


## Drawbacks

- Self attention takes quadratic time and space (each token attends to all other tokens); Scaling them becomes an issue

- This is tackled in subsequent work on Transformers: Big Bird, Linformer, Reformer

## Applications

#### GPT-3
- consists of only the decoder blocks from transformers
- Pretrained on a Language Modelling task: 
- For downstream tasks:
	- Either train a classifier on the last hidden state (e.g. NLI, Sentiment Classification)
	- Use the pretrained network as a generator (e.g. Summerization, NLG)

- One major success of GPT-3 had been able to perform extremely well under few-shot settings without performing gradient updates ("in-context" learning)

#### BERT
- Consists of only the encoder blocks from transformers
- Can't be pre-trained on a naive language modelling task (data "leakage" from future given it gets bidirectional context)
- Clever idea: Replace certain words with a placeholder and predict (Masked Language Modeling)
- Also makes use of another training objective given two chunks of text predict whether the second chunk follows the other or not(Next Sentence Prediction)
- For downstream tasks, fine tune with an additional classification layer