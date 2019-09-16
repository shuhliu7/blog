# Bert

Bert is google's new technique for NLP pre-training which called Bidirectional Encoder Representations from Transformers. 

## The powerful of Bert
- Bidirectional

- pretraining

- self-attention

####Bidirectional
Compare with privious model, Bert achieves the goal that changing sequential computation to bidirectional. It can help us to overcome the restriction in left-to-right architecture that every token can only attended to previous tokens in the self-attention layers of the Transformer([Vaswani et al.,2017](https://pdfs.semanticscholar.org/8656/df5ece8f482c717e8381cc114dee161f9a3f.pdf?_ga=2.91985625.233109167.1566939361-747780848.1566326833)). Also, Bert proposing mask language model to pre-train bidirectional representaion. Next sentence pretiction is also the task bert has been pretrained on.

####Pretraining
One of the biggest task of machines learning is obtaining enough data. Bert uses large amount of data on pre-training, which help us to weaken this problem. Fine-tuning is a friendly mechnism to small company and individuality. 

####Self-attention
I think Self-attention layer is the most important part in Bert. BERT uses the self-attention mechanism to encoding a concatenated text pair between two sentences. For multi-head attention, it is the core of the transformer architecture that transforms hidden states for each element of a sequence based on the other elements. 

``` {code in Google-Bert}
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
```
  Performs multi-headed attention from `from_tensor` to `to_tensor`.
  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.
  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].
  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.
  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.
  ```
  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).
  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  ```
