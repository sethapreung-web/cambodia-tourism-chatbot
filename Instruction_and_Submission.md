# Final Project: Tourism Chatbot Assistant (Cambodia)

## Project Overview

In this project, you will build an **end-to-end AI chatbot system** that helps users with tourism-related questions about Cambodia.

You will:

* Train a **SimpleRNN model**
* Build a **chatbot application using Streamlit**
* Deploy your app on **Streamlit Cloud**

---

## Objective

The goal is NOT to build a perfect chatbot, but to:

* Understand how **RNN processes sequences**
* Build a **complete AI pipeline (training → inference → deployment)**
* Analyze the **limitations of SimpleRNN**

---

## Problem Statement

> Build a chatbot that can answer tourism-related questions about Cambodia.

---

## Example Use Cases

Your chatbot should be able to answer questions like:

* “Where is Angkor Wat?”
* “What food should I try in Cambodia?”
* “Best time to visit Cambodia?”
* “Is Cambodia safe?”
* “How to travel in Phnom Penh?”

---

## Project Requirements

## 1. Model (MANDATORY)

* Must use **SimpleRNN**
* Must include:
  * Tokenization
  * Padding
  * Training

---

## 2. Notebooks

### `train.ipynb`

* Load dataset
* Preprocess data
* Train model
* Save:

  * `model.h5`
  * `tokenizer.pkl`
  * config (e.g., `max_len`)

---

### `prediction.ipynb`

* Load trained model
* Test at least **5 inputs**
* Show:
  * Input
  * Output
* Perform **error analysis**

---

## 3. Streamlit App (`app.py`)

Your app must:

* Accept user input
* Display chatbot response
* Support **multiple conversations (chat history)**

---

## 4. Deployment (MANDATORY)

* Deploy your app on **Streamlit Cloud**
* Submit working link
* GitHub Repository URL or Zip file (If you have problem running with python)

---

# Project Structure

```bash
project/
│
├── app.py
├── model.h5
├── tokenizer.pkl
├── config.pkl #(if your project has)
├── train.ipynb
├── prediction.ipynb
├── requirements.txt
```

---

# Deliverables

## 1. Code Submission

* All project files

## 2. Report (1–2 pages)

Include:

* Problem description
* Model architecture
* Results
* Error analysis

---

## Grading Rubric

| Component                        | Weight |
| -------------------------------- | ------ |
| Model Implementation (SimpleRNN) | 35%    |
| Streamlit App                    | 25%    |
| Deployment                       | 15%    |
| Evaluation & Error Analysis      | 25%    |

---

### Important Notes

* Your chatbot is **NOT expected to be perfect**
* You will NOT get full marks for just running the code
* Your **analysis and understanding** are critical

---

### Key Question You Must Answer

In your report:

> “When does your chatbot fail, and why?”

---
