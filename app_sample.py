import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Load artifacts
# ==============================

model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]


# ==============================
# Response function
# ==============================

def generate_response(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = model.predict(seq, verbose=0)
    pred = np.argmax(pred, axis=-1)[0]

    result = []
    for idx in pred:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                result.append(word)
                break

    return " ".join(result)


# ==============================
# Streamlit UI
# ==============================

st.title("SimpleRNN Chatbot 🤖")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
user_input = st.text_input("You:", key="input")

# Send button
if st.button("Send"):
    if user_input.strip() != "":
        bot_response = generate_response(user_input)

        # Save to history
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_response))

# Display chat history
st.subheader("Conversation")

for speaker, text in st.session_state.history:
    if speaker == "You":
        st.write(f"🧑 You: {text}")
    else:
        st.write(f"🤖 Bot: {text}")

# ==============================
# End / Clear Chat
# ==============================

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []

with col2:
    if st.button("❌ End Chat"):
        st.session_state.history = []
        st.write("Chat ended. Refresh to start again.")