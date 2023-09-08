import streamlit as st 
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

model = tf.keras.models.load_model('model.h5')

st.title("上傳圖片(0~9)辨識")

uploaded_file = st.file_uploader("上傳圖片(.png)", type="png")
if uploaded_file is not None:
    image1 = io.imread(uploaded_file, as_gray=True)
    image_resized = resize(image1, (28, 28), anti_aliasing=True)    
    X1 = image_resized.reshape(1,28, 28) #/ 255
    X1 = np.abs(1-X1)
    st.write("predict...")
    predictions = np.argmax(model.predict(X1), axis=-1)
    st.markdown(f"# {predictions[0]}")
    st.image(image1)
