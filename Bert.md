# Bert

Bert is google's new technique for NLP pre-training which called Bidirectional Encoder Representations from Transformers. 

## The powerful of Bert
- Bidirectional

- pretraining

- self-attention

##Bidirectional
Compare with privious model, Bert achieves the goal that changing sequential computation to bidirectional. It can help us to overcome the restriction in left-to-right architecture that every token can only attended to previous tokens in the self-attention layers of the Transformer([Vaswani et al.,2017](https://pdfs.semanticscholar.org/8656/df5ece8f482c717e8381cc114dee161f9a3f.pdf?_ga=2.91985625.233109167.1566939361-747780848.1566326833)). Also, Bert proposing mask language model to pre-train bidirectional representaion. Next sentence pretiction is also the task bert has been pretrained on.

###pretraining
One of the biggest task of machines learning is obtaining enough data. Bert uses large amount of data on pre-training, which help us to weaken this problem. Fine-tuning is a friendly mechnism to small company and individuality. 

###self-attention
Self-attention layer is one of the most important part in Bert. BERT uses the self-attention mechanism to encoding a concatenated text pair between two sentences. For multi-head attention, it is the core of the transformer architecture that transforms hidden states for each element of a sequence based on the other elements. The multi-head layer, consists of n different dot-product attention mechanisms. At a high level, attention represents a sequence element with a weighted sum of the hidden states of all the sequence. In multi-head attention the weights in the sum us dot product similarity between transformed hidden states. Concretely, the ith attention mechanism ‘head’ is:
![Image description](/Users/huanranyixin/blog/屏幕快照 2019-09-02 23.23.54.png)
