# Cambodia Tourism Chatbot

A simple chatbot that answers questions about Cambodia tourism, built with TensorFlow/Keras and deployed using Streamlit. The model uses a SimpleRNN architecture trained on around 2000 question-answer pairs.

## Demo

Live app: [link to your Streamlit app - fill in after deployment]

## Tech Stack

- Python 3.11
- TensorFlow / Keras (SimpleRNN)
- Streamlit
- Pandas, NumPy, Scikit-learn

## How to Run Locally

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

## Files

- `train.ipynb` - training notebook
- `prediction.ipynb` - prediction and error analysis
- `app.py` - Streamlit chatbot interface
- `model.h5` - trained SimpleRNN model
- `tokenizer.pkl` - fitted tokenizer
- `config.pkl` - model configuration
- `requirements.txt` - Python dependencies
- `cambodia_tourism_dataset_large.csv` - training dataset

## Notes

This project was built for an academic course on deep learning. The chatbot has known limitations (repetition, struggles with long inputs, no topic boundaries) which are discussed in the project report. These are characteristic of vanilla SimpleRNN architectures.
