import numpy as np
import streamlit as st
import cv2
from keras.models import load_model




model = load_model('C:/ANKITH/AI ML/PROJECTS/8 Plant_Disease_Detection/plant_disease.h5')


CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot',
               'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Potato___Early_blight', 'Potato___Late_blight','Potato___healthy']


st.title("PLANT DISEASE DETECTION")
st.markdown("Unput an image of the plant leaf")


plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')



if submit:

    if plant_image is not None:

        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        opencv_image = cv2.resize(opencv_image, (256,256))

        opencv_image.shape = (1,256,256,3)

        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('___')[0]+ " leaf with " + result.split('___')[1]))