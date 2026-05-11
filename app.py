# =====================================================
# Project: Cambodian Tourism Chatbot - Streamlit App
# =====================================================
#
# This Streamlit app loads the trained SimpleRNN model from train.ipynb
# and provides a chat interface so users can ask questions about Cambodia
# tourism. The chatbot uses the same prediction logic as prediction.ipynb.
# =====================================================

# 1. Import libraries
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================
# 2. Page Configuration
# =====================================================
st.set_page_config(
    page_title="Cambodian Tourism Chatbot",
    layout="centered",
)


# =====================================================
# 3. Load Model and Artifacts
# =====================================================
# The @st.cache_resource decorator makes sure the model is loaded only
# once, not every time the user sends a message.

@st.cache_resource
def load_artifacts():
    # Load the trained model
    model = load_model("model.h5")

    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load the configuration
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    # Build the reverse mapping (index -> word)
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    return model, tokenizer, config, index_to_word


model, tokenizer, config, index_to_word = load_artifacts()
max_len = config["max_len"]


# =====================================================
# 4. Prediction Function
# =====================================================
# This function takes a user input string, processes it, and returns the
# model's predicted response. It uses the same logic as prediction.ipynb.

def predict_answer(text):
    # Step 1: Convert input text to lowercase and tokenize
    seq = tokenizer.texts_to_sequences([text.lower()])

    # Step 2: Pad the sequence to max_len
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    # Step 3: Get prediction from the model
    pred = model.predict(seq, verbose=0)

    # Step 4: Take the most likely word at each position (greedy decoding)
    pred_ids = np.argmax(pred, axis=-1)[0]

    # Step 5: Convert integers back to words, skip padding (0)
    words = []
    for idx in pred_ids:
        if idx == 0:
            continue
        word = index_to_word.get(idx, "")
        if word:
            words.append(word)

    return " ".join(words) if words else "(no response)"


# =====================================================
# 5. Main User Interface
# =====================================================
st.title("Cambodian Tourism Chatbot")
st.caption(
    "Ask me questions about Cambodia tourism such as Angkor Wat, "
    "Cambodian food, currency, or travel tips."
)

# Initialize chat history in session state so it persists across reruns
if "history" not in st.session_state:
    st.session_state.history = []

# Display the existing chat history
for speaker, text in st.session_state.history:
    if speaker == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)

# Chat input box at the bottom of the page
user_input = st.chat_input("Type your question here...")

if user_input:
    # Save and display the user's message
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display the chatbot's response
    bot_response = predict_answer(user_input)
    st.session_state.history.append(("bot", bot_response))
    with st.chat_message("assistant"):
        st.write(bot_response)


# =====================================================
# 6. Sidebar - Information and Examples
# =====================================================
with st.sidebar:
    st.header("About this Chatbot")
    st.write(
        "This chatbot was built as a final project for an academic course "
        "on deep learning. The model uses a SimpleRNN architecture trained "
        "on around 2000 Cambodia tourism question-answer pairs."
    )

    st.subheader("Try asking me")
    st.write("- Where is Angkor Wat?")
    st.write("- What food should I try in Cambodia?")
    st.write("- What currency is used in Cambodia?")
    st.write("- When is the best time to visit?")
    st.write("- What is the capital of Cambodia?")

    st.subheader("Known Limitations")
    st.write(
        "The chatbot uses a SimpleRNN, which has known limitations:"
    )
    st.write("- It often repeats the same word")
    st.write("- It struggles with long or off-topic questions")
    st.write("- It cannot handle words outside its training vocabulary")
    st.write(
        "These limitations are discussed in detail in the project report."
    )

    st.divider()

    # Button to clear the chat history
    if st.button("Clear chat history"):
        st.session_state.history = []
        st.rerun()
