# Cambodia Tourism Chatbot — Project Report

**Student:** Setha Proeung
**Date:** May 2026

## Problem Description

This project builds a chatbot that answers questions about Cambodia tourism using a SimpleRNN model. The dataset (`cambodia_tourism_dataset_large.csv`) contains 2000 question-and-answer pairs covering destinations, food, currency, and travel logistics. The goal is not only to build a working chatbot but also to understand the limitations of vanilla SimpleRNN for sequence-to-sequence tasks.

## Model Architecture

The model is a Sequential Keras model with three layers:

- **Embedding** (`input_dim=vocab_size`, `output_dim=64`, `mask_zero=True`) — turns each word index into a 64-dimensional vector.
- **SimpleRNN** (128 units, `return_sequences=True`) — processes the sequence and outputs one hidden state per position.
- **Dense** (`vocab_size`, softmax) — outputs a probability over all words at each position.

Preprocessing: tokenization with Keras `Tokenizer(filters='')`, padding with `pad_sequences(padding='post')`, one-hot encoded targets, and an 80/20 train-test split. The model was compiled with Adam optimizer, `categorical_crossentropy` loss, and trained for 100 epochs with batch size 32.

## Results

After training, the model reached:
- Training accuracy: **64.3%**
- Validation accuracy: **58.8%**

The training and validation loss curves showed clear **overfitting**: training loss kept decreasing while validation loss steadily increased from 1.60 to 1.78. Both accuracies plateaued within the first few epochs, meaning extra training only memorized the training data instead of learning to generalize.

Sample outputs from the trained chatbot:

| Question | Response |
|---|---|
| Where is Angkor Wat? | it is is in in in in in in in in in |
| What food should I try in Cambodia? | it should try amok and lok lak lak lak lak |
| What currency is used in Cambodia? | it is and and and widely widely widely widely |

The model correctly recalls real Cambodia facts (amok, lok lak, the riel currency context) but cannot finish a sentence cleanly.

## Error Analysis

The chatbot fails in several predictable ways:

1. **Repetition collapse** — outputs get stuck on a single word (e.g. `lak lak lak`). Caused by greedy `argmax` decoding combined with SimpleRNN's vanishing gradients, which prevent the model from remembering its own previous outputs.
2. **No topic boundary** — off-topic questions like *"How do I bake a cake?"* still produce Cambodia tourism vocabulary, because that is the only vocabulary the model knows.
3. **Long input failure** — questions of 30+ words produce extremely short responses, because early tokens are lost to the vanishing gradient.
4. **Short input failure** — single-word inputs like *"food"* collapse into repetition, because most of the padded sequence is zeros.
5. **Misleading accuracy** — 59% validation accuracy looks decent, but it is inflated by correctly predicting padding tokens and common words like `is`, `the`, `and`.

## Chatbot Limitations

The failures above are not bugs in the code; they are well-known limitations of the SimpleRNN architecture. SimpleRNN has no gating mechanism to control memory, no attention to focus on relevant input tokens, and no way to handle out-of-vocabulary words. To improve this chatbot, the recommended next steps are to replace SimpleRNN with **LSTM** or **GRU**, add **Dropout**, use **early stopping**, try **beam search** instead of greedy decoding, and ideally move to a **Transformer with attention**.

---

**Links:** GitHub: [https://github.com/sethapreung-web/cambodia-tourism-chatbot] · Streamlit: [https://cambodia-tourism-chatbot-9ajzkgi9ghr2myvelt8zbv.streamlit.app/]
