import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import base64
st.title("MNIST Classification")
image_path = '9 kb sign (1).jpeg'

#Convert the image to base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

col3 , col4 = st.columns(2)

# Construct the absolute path of the model file
model_path = r"/Users/aditighosh/Downloads/MNNIST-Classification--main/mnist_model.h5"
# abs_model_path = os.path.abspath(model_path)

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please make sure the model file is in the correct directory.")
else:
    try:
        # Load the model
        model = load_model(model_path)
        st.success(f"Model loaded successfully from {model_path}.")
        
        # Divide into two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Draw your digit here:")
            canvas_result = st_canvas(
                fill_color='#000000',
                stroke_width=30,
                stroke_color='#FFFFFF',
                background_color='#000000',
                width=300,
                height=300,
                drawing_mode="freedraw" if st.checkbox("Draw (or Delete)?", True) else "transform",
                key='canvas'
            )
        
        with col2:
            if canvas_result.image_data is not None:
                img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
                rescaled = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
                st.write("Your drawn image: ")
                st.write('Model Input')
                st.write(' ')
                st.image(rescaled)

                # Convert the image to grayscale and normalize
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = gray_img / 255.0

                # Add a batch dimension
                img_array = gray_img.reshape((1, 28, 28, 1))

                # Make a prediction
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)

                # Display the prediction
                st.write("Predicted Class:", predicted_class)
                image_html = f'''
                <div style="text-align: right;  padding-right: 15px;">
                    <img src="data:image/jpeg;base64,{encoded_image}" alt="Image" width="150">
                </div>
                '''
                st.markdown(image_html, unsafe_allow_html=True)
            else:
                st.write("Please draw a digit on the canvas.")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
