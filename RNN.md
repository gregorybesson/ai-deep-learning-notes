# Recurrent Neural Networks (RNN)

Feed-forward networks (the input is fed into the network and propagates through the hidden layers to the output layer.
There is no sense of order in the inputs.

With RNN, output data feed input data so that we have a sense of order

RNN are able to remember past values so that they can improve their prediction. 

regular RNN can't have a Long Term memory because weights of initial inputs are vanishing during forward and back propagations.

Improved cells called LSTM give this ability to improve the memory of RNN.

Long Short Term Memory (LSTM)





## Sequence to sequence
For a chatbot, we'd like to read a sequence of any length and ouput sequences of any length.
For that, we'll use 2 RNN: 1 for the input sequence (encoder), the other for the ouput sequence (decoder)

Application: 
- Translation
- summarize texts
- Question-Answering model

### Architecture
Input => RNN-Encoder => context (always the same size) => RNN-Decoder => Output

Retrieval-based models: Trained to understand input text and propose answers from pre-existing responses.
Easy to do, good for customer services.
Generative-based models: They generate new responses. Hard to do.  The input quality is key here to generate goor responses.

Sequence to sequence is naturally about generative-based models.

Keywords:
- <EOS> : end of output sequence 
- <UNK> : a token we don't want to learn (ie. the names) is replaced with this keyword during embedding
- <PAD> : keyword used to complete an input sequence when the input is too short.
- <GO> : This is the inpput to the first time step of the decoder to let him know when to start generating the output.

### Tensorflow
We'll use 
- tf.nn
- tf.contrib.rnn
- tf.contrib.seq2seq

Encoder: this is a ```tf.nn.dynamic_rnn```
Decoder: this is a ```tf.contrib.seq2seq.dynamic_rnn_decoder```

Inputs : ```tf.nn.embedding_lookup``` to turn words into vectors

Process:
- we build our vocabulary of unique words (and count the occurrences while we're at it)
- we replace words with low frequency with <UNK>
- create a copy of conversations with the words replaced by their IDs
- we can choose to the <GO> and <EOS> word ids to the target dataset now, or do it on training time






