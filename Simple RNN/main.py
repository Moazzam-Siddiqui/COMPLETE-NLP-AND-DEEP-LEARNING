##Import all the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

#Load the imdb dataset and word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#Load the pretrained model
model = load_model('simple_rnn.h5')

#Step 2:Helper Function
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

##Design Streamlit App
import streamlit as st

st.title('IMDB Movie review Sentiment Analysis')
st.write('Enter a movie review to classify it as negative and positive')

#User Input
usr_input = st.text_area('Movie Review')

# Add test buttons for debugging
col1, col2 = st.columns(2)
with col1:
    if st.button('Test with "bad movie"'):
        usr_input = "bad movie"
with col2:
    if st.button('Test with "terrible awful horrible"'):
        usr_input = "terrible awful horrible boring waste"

if st.button('Classify'):
    if usr_input.strip():
        preprocessed_input = preprocess_text(usr_input)
        
        #Make Prediction
        prediction = model.predict(preprocessed_input, verbose=0)
        
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        
        #Display the Result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]:.6f}')
        
        # Debug information
        st.write("---")
        st.write("**Debug Information:**")
        words = usr_input.lower().split()
        st.write(f"Words: {words}")
        
        # Check which words are found in vocabulary
        found_words = []
        unknown_words = []
        for word in words:
            if word in word_index:
                found_words.append(word)
            else:
                unknown_words.append(word)
                
        st.write(f"Found in vocabulary: {found_words}")
        st.write(f"Unknown words: {unknown_words}")
        st.write(f"Encoded sequence sample: {preprocessed_input[0][:20]}")
        
    else:
        st.write("Please Enter a Movie Review.")
