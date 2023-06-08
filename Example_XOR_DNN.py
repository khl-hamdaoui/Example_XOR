import numpy as np
import keras
import streamlit as st
from keras.models import Sequential
from keras.layers.core import Dense
import requests

# Download the model file from GitHub
url = 'https://github.com/khl-hamdaoui/Example_XOR/raw/main/keras_model.h5'
response = requests.get(url)
with open('keras_model.h5', 'wb') as f:
    f.write(response.content)

# Load the Keras model from the local file
loaded_model = keras.models.load_model('keras_model.h5')
# Load the Keras model
#loaded_model = keras.models.load_model('https://github.com/khl-hamdaoui/Example_XOR/blob/main/keras_model.h5')
# creating a function for Prediction
def XOR(input_data):   
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = np.round(loaded_model.predict(input_data_reshaped))
    print(prediction)
    if (prediction[0] == 0):
      return 'False'
    else:
      return 'True' 
def main():   
    # giving a title
    st.title('XOR Function Web App with DNN')
    # getting the input data from the user
    First_value = st.number_input('First Value')
    Second_value = st.number_input('Second Value')
    # code for Prediction
    test = ''  
    # creating a button for Prediction
    if st.button('Result'):
        test = XOR([First_value, Second_value]) 
    st.success(test)
if __name__ == '__main__':
    main()
    
    
