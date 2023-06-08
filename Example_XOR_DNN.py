import numpy as np
import keras
import streamlit as st

# Load the Keras model
loaded_model = keras.models.load_model('keras_model.h5')
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
    
    