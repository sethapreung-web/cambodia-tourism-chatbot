"""
Cambodia Tourism Chatbot - Streamlit App

A simple chatbot interface that loads the trained SimpleRNN model and
allows users to chat with it through a web interface.
"""

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Cambodia Tourism Chatbot",
    page_icon="KH",
    layout="centered",
)


# =====================================================
# Load Model and Artifacts (cached so it only runs once)
# =====================================================
@st.cache_resource
def load_artifacts():
    """Load the trained model, tokenizer, and config."""
    model = load_model("model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    return model, tokenizer, config, index_to_word


model, tokenizer, config, index_to_word = load_artifacts()
max_len = config["max_len"]


# =====================================================
# Response Generation Function
# =====================================================
def generate_response(text):
    """Take a user question and return the chatbot's response."""
    seq = tokenizer.texts_to_sequences([text.lower()])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    pred = model.predict(seq, verbose=0)
    pred_ids = np.argmax(pred, axis=-1)[0]

    words = []
    for idx in pred_ids:
        if idx == 0:
            continue
        word = index_to_word.get(idx, "")
        if word and word != "<OOV>":
            words.append(word)

    return " ".join(words) if words else "Sorry, I do not understand."


# =====================================================
# Main UI
# =====================================================
st.title("Cambodia Tourism Chatbot")
st.caption(
    "Ask me about Angkor Wat, Cambodian food, currency, travel tips, "
    "and more. I am a SimpleRNN model trained on a small dataset, "
    "so my answers are not always perfect."
)

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Display existing chat history
for speaker, text in st.session_state.history:
    if speaker == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)

# Chat input box at the bottom
user_input = st.chat_input("Ask a question about Cambodia...")

if user_input:
    # Save and display the user's message
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display the bot's response
    bot_response = generate_response(user_input)
    st.session_state.history.append(("bot", bot_response))
    with st.chat_message("assistant"):
        st.write(bot_response)


# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot was built as a final project for a deep learning "
        "course. It uses a SimpleRNN model trained on about 2000 "
        "Cambodia tourism question-answer pairs."
    )

    st.subheader("Try asking me")
    st.write("- Where is Angkor Wat?")
    st.write("- What food should I try in Cambodia?")
    st.write("- What currency is used in Cambodia?")
    st.write("- When is the best time to visit?")
    st.write("- What is the capital of Cambodia?")

    st.subheader("Limitations")
    st.write(
        "The model often gets stuck repeating words, struggles with "
        "long or off-topic questions, and cannot handle words that "
        "were not in its training vocabulary. These are known "
        "limitations of the SimpleRNN architecture."
    )

    st.divider()

    if st.button("Clear chat history"):
        st.session_state.history = []
        st.rerun()
