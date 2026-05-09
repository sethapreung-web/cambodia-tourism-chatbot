# Cambodia Tourism Chatbot — Final Project Report

**Course:** [Your Course Name]
**Student:** Setha Proeung
**Date:** May 2026

---

## 1. Problem Description

For this project, I built a chatbot that answers questions about Cambodia tourism. The chatbot should be able to take a question from the user (for example, "Where is Angkor Wat?" or "What food should I try in Cambodia?") and reply with a relevant answer.

I used a SimpleRNN model from TensorFlow Keras, as required by the assignment. The dataset I used is `cambodia_tourism_dataset_large.csv`, which has 2000 question and answer pairs. Each row in the dataset has one question in the `question` column and one answer in the `answer` column.

The point of this project is not really to build a perfect chatbot, but to learn how SimpleRNN works, what it can do, and where it fails. So in this report, I focus a lot on showing where the chatbot does not work well and explaining why that happens.

---

## 2. Model Architecture

The model has three layers:

1. **Embedding layer** — turns each word (which is just an integer index after tokenization) into a vector of 64 numbers. This is so the model can learn meanings of words.
2. **SimpleRNN layer** — has 128 units and uses `return_sequences=True` so it gives one output for every position in the sequence (not just the last one). This is needed because we want to predict a full sentence, not just one word.
3. **Dense layer** — has size equal to the vocabulary, with a softmax activation. It gives a probability for each word in the vocabulary at every position.

For the loss function, I used `sparse_categorical_crossentropy` instead of `categorical_crossentropy`. This is because my targets are integer sequences, not one-hot vectors, and using the sparse version saves a lot of memory. With 2000 samples and a vocabulary of around 1000 words, one-hot encoding would create a very large matrix and probably crash my laptop.

The optimizer is Adam, and I tracked accuracy as the metric.

For preprocessing:
- I used Keras `Tokenizer` with an `<OOV>` token to handle unknown words.
- I padded all sequences to a fixed length (`max_len`) which I set to the 95th percentile of all sentence lengths. This way I don't waste memory on a few very long outliers.
- I split the data into 80% training and 20% validation, with `random_state=42` so the results are reproducible.

---

## 3. Results

I trained the model for 500 epochs with batch size 32. Below are the main results.

### 3.1 Training Numbers

After 500 epochs, the final accuracy was:

- Training accuracy: **64.29%**
- Validation accuracy: **58.80%**

The gap between training and validation is small (about 5%), which at first looked good. But when I looked at the loss curves, I could see overfitting clearly.

### 3.2 Training Curves

I plotted training loss, validation loss, training accuracy, and validation accuracy over the 500 epochs.

What I noticed:
- The training loss went down very slowly (from about 1.17 to 1.15).
- The validation loss actually went UP, from about 1.60 to 1.78.
- Both accuracies stayed almost flat the entire time, basically from epoch 1.

This means the model basically learned everything it could in the first few epochs, and then more training only made it overfit. The validation loss going up while training loss goes down is the classic sign of overfitting.

If I had to do this again, I would use early stopping based on validation loss. The best model was probably around epoch 5 to 10, not epoch 500.

### 3.3 Sample Outputs

Here are some example responses from the trained chatbot:

| Question | Response |
|---|---|
| Where is Angkor Wat? | it is is and and and and and and and and and |
| What is the capital of Cambodia? | it is gmt plus the |
| What currency is used in Cambodia? | it is is riel local |
| What food should I try in Cambodia? | it should try amok and lok lak lak lak lak lak lak |
| When is the best time to visit Cambodia? | it is usually sunrise is the best |
| Tell me about cryptocurrency in Cambodia | it is is the but |
| Tell me about Cambodian cuisine | it is is the but but but but but but but but |
| I am planning a trip to Cambodia next month... | of the pepper |
| food | lok lok lok lok lok lok lok lok lok lok lok lok |
| How do I bake a chocolate cake? | passapp can take a visa night or or or or or or |

Some of these are surprisingly correct. The model knows that the currency is "riel", that "amok" and "lok lak" are Cambodian dishes, and that the country is in GMT timezone. But the model also has serious problems, which I explain in the next section.

---

## 4. Error Analysis — When Does the Chatbot Fail and Why?

This is the part I think is the most important. I tested the chatbot on several different categories of input and found that it fails in several specific ways.

### 4.1 Repetition Collapse

This is the most obvious failure. The model often gets stuck repeating the same word over and over.

Examples:
- "and and and and and and..."
- "lak lak lak lak lak lak..."
- "but but but but but but..."
- "or or or or or or..."

**Why does this happen?**

Two reasons combine:

First, my prediction code uses **greedy decoding** with `np.argmax`. At every position, the model picks the single word with the highest probability. This is the simplest decoding method but it has no diversity.

Second, **SimpleRNN has the vanishing gradient problem**. The hidden state cannot remember information from many steps ago. So when the model just predicted "lak", by the next step it has already started to forget that. So "lak" is still the best choice, and so on.

This is exactly why people invented LSTM and GRU — they have gates that help the network remember things over long sequences. SimpleRNN just doesn't have that.

### 4.2 Topic Boundary Failure

The training data is only about Cambodia tourism. But the model doesn't know that. When I asked it something completely off-topic:

- Question: "How do I bake a chocolate cake?"
- Answer: "passapp can take a visa night or or or"

The model just replied with words from its Cambodia vocabulary ("Passapp" is actually the name of a Cambodian ride-hailing app, and "visa" is a travel word). It has no idea the question is about cake. There is no concept of "I don't know" in the model.

### 4.3 Long Input Failure

When I asked a long question:

- Question: "I am planning a trip to Cambodia next month with my family and we have never been to Southeast Asia before, so could you tell me what the most important things to know about Cambodian culture are?"
- Answer: "of the pepper"

The response is only 3 words. This is also caused by the vanishing gradient. By the time the SimpleRNN reads the end of the long question, it has basically forgotten the beginning. So it's only really conditioning on the last few words ("Cambodian culture are"), and even that signal is weak.

### 4.4 Short Input Failure

The opposite problem also happens:

- Question: "food"
- Answer: "lok lok lok lok lok lok lok..."

When the input is just one word, most of the padded sequence is zeros. The model has very little to work with, so it falls back on a memorized food word ("lok") and gets stuck repeating it.

### 4.5 Misleading Accuracy

The validation accuracy was 59%, which sounds okay. But the actual responses are clearly bad. Why is there a gap?

I think this is because the accuracy metric counts every position in the padded sequence. Most positions are padding (zeros), and the model can predict zeros easily. Common words like "is", "the", "and" also appear a lot and are easy to predict. So the accuracy gets inflated by these easy predictions, while the actual content words (the meaningful ones) are wrong.

This taught me that accuracy is not a great metric for sequence generation tasks. For real evaluation, you would need something like BLEU score, or just look at the outputs by hand.

### 4.6 Out-of-Vocabulary Words

When I used words that were not in the training data (like "cryptocurrency"), the tokenizer replaced them with the `<OOV>` token. This means the model sees the same input no matter what unknown word I used. So the response cannot really depend on what I actually said.

---

## 5. What I Learned

Doing this project taught me some things that I don't think I would have learned just from reading about RNNs:

1. **The architecture matters more than the training time.** I tried 100, 250, and 500 epochs, and the responses were basically the same. SimpleRNN has a ceiling, and no amount of extra training can break through it.

2. **Accuracy can lie.** A model with 59% accuracy can still produce nonsense. You always have to look at the actual outputs, not just the numbers.

3. **Greedy decoding is too simple.** Using argmax at every step causes repetition. Better methods like beam search or temperature sampling would probably help.

4. **SimpleRNN is mostly outdated.** All modern NLP uses Transformers (like BERT or GPT) because they have attention mechanisms that solve most of these problems. LSTM and GRU are also much better than SimpleRNN.

---

## 6. How to Improve

If I had more time, I would try the following:

- Replace SimpleRNN with **LSTM** or **GRU** to fix the vanishing gradient issue.
- Add **dropout** to reduce overfitting.
- Use **early stopping** so I don't train past the best epoch.
- Use **beam search** instead of greedy decoding to reduce repetition.
- Try a small **Transformer** with attention.
- Get more training data (2000 pairs is small).
- Use a **pretrained word embedding** like GloVe or fastText instead of training embeddings from scratch.

---

## 7. Conclusion

The chatbot works in the sense that it produces some response for any input, and it has even memorized some real Cambodia facts (riel, amok, lok lak, GMT timezone). But it fails in many predictable ways: repetition collapse, no topic boundaries, struggles with long inputs, struggles with short inputs.

These failures are not bugs in my code. They are well-known limitations of the SimpleRNN architecture, which is exactly why researchers moved on to LSTM, GRU, and Transformers. Doing this project gave me a much better understanding of why those architectures exist and what problems they were designed to solve.

---

## 8. Files Submitted

- `train.ipynb` — training notebook
- `prediction.ipynb` — prediction and error analysis notebook
- `app.py` — Streamlit chatbot app
- `model.h5` — trained model weights
- `tokenizer.pkl` — fitted tokenizer
- `config.pkl` — configuration (max_len, vocab_size)
- `requirements.txt` — Python dependencies
- `README.md` — project description
- `cambodia_tourism_dataset_large.csv` — training dataset

**GitHub repo:** [link]
**Streamlit deployment:** [link]
