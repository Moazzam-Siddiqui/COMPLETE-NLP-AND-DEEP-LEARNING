import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the Model

model = load_model('Next_Word_LSTM.h5')

#Load the Tokenizer

with open('token.pickle','rb') as handle:
    token = pickle.load(handle)
    

# Function to Predict the Next Word
def predict_next_word(model,token, text,max_sequence_len):
    token_list = token.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # Ensures the sequence length matches the max_sequences_len-1
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis =1)
    for word ,index in token.word_index.items():
        if index == predicted_word_index:
            return word
    return None
    

#Streamlit App

st.title("NEXT WORD PREDICTOR")
input_text = st.text_input("Choose your Words Carefully","to be or not to be")


if st.button("PREDICT"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,token,input_text,max_sequence_len)
    st.write(f"Next word will be: {next_word}")