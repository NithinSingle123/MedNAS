
- First came RNNs which were fine but the sentences generated were outlandish of sorts
- then came LSTMs where in the flow of a natural sentence started appearing
- Then came GPT 2 with 1 billion parameters and it was able to generate pretty convincing sentences
- Then came GPT 3 with 175 billion parameters and the sentences it generates are so realistic that the humans are not able to distinguish it with human written news articles.

## A History of Language Modeling at OpenAI

Why Unsupervised Learning?
Supervised learning works great and comes with guarantees! But large labelled datasets are hard to find

Can we also learn from Unlabeled data
--> Unsupervised Learning

the approach is to use internet as this large trove of unlabeled data

##### Why Use Autoregressive Generative Models for Unsupervised Learning

- Doing very well at next-token prediction requires more than modeling local correlations, and perhaps even some "reasoning"
- first sign of life --> 2017 instead of training a classifier model OPEN AI trained an LSTM model to predict the next character in Amazon Reviews and when they trained a linear model on the features surprisingly, they found one of these cells or one of these neurons was firing in terms of predicting sentiment, positive activations for positive reviews and negative activations for negative reviews and this is despite not seeing any labels at training time.

![[Screenshot 2026-07-23 045416.png]]

- GPT 1 was trained on the internet and not on amazon reviews anymore and it was fine tuned on a bunch of different downstream tasks.

- Like GPT 1, GPT 2 was trained on a large chunk of the internet and its only trained to predict the next token or word from previous words but the key insight of GPT 2 is that many  downstream tasks can be expressed naturally as a language model in the past.
- GPT 2 explores how well we can perform on these downstream tasks without the need of fine tuning by using the above method.

#### GPT-3: Language Model MetaLearning

![[Pasted image 20260723050506.png]]

##### GPT-3: Few Shot Arithmetic

- You are taking the entire context slide of your transformer and you are putting in as many examples as will fit and then finally you put in the example that you would like to solve.

![[Pasted image 20260723050807.png]]

- so here you can see as the model gets larger the better it is able to recognize the task whether it is additions or subtraction.
- Same with the case of word unscrambling

### Autoregressive Language Modeling is Universal

- It is not mandatory that transformers have to be applied to languages only, transformer models are sequential and and as such it can just ingest any sequence of bytes and model them.

> INDUCTIVE BIASES


- So we can also think if that is the case then can we do the same with audio and video

#### Can we apply GPT to images?

![[Pasted image 20260723051638.png]]

#### Codex

- A code writing model
- Isn't code just another modality? Why is it worth the effort to train a model on code ?

- GPT 3 had a rudimentary ability to write Python code from a docstring or method name, even though there was little code in the training data.
- Functions can be tested with unit tests and an interpreter unlike texts where a human is required.

- Needed a sandbox to avoid external malicious code

![[Pasted image 20260723053220.png]]

#### Codex Training Details

Dataset: 159GB of code collected from 54 million repositories

For efficient training: fine-tuned from GPT-3 models of different sizes

Extra spaces in tokenizer that compress runs of white spaces so that the code doesn't get split into a lot of individual tokens (makes training 30 to 40 percent efficient)

![[Pasted image 20260723054202.png]]

![[Pasted image 20260723054344.png]]

![[Pasted image 20260723054447.png]]

#### Conclusion

- Progress in neural language modeling has been rapid
- GPT  was the result of work on unsupervised learning in language.
- Autoregressive modeling is universal, and yields strong results in image and text-to-image modeling
- We can produce strong code generating models by fine tuning GPT-3 on code. Sampling is an unreasonably effective way to improve model performance.



