import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

def load_and_prepare_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
    return img

def predict(img):
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    model = 'covid_model_v4.h5'  # Ensure the model file is in the same directory
    model = load_model(model)

    pred = model.predict(img)

    if np.argmax(pred, axis=1)[0] == 1:
        out_pred = 'You Are Safe, But Do keep precaution'
    else:
        out_pred = 'You may have Coronavirus, Get yourself Tested...'

    return out_pred, float(np.max(pred))

def main():
    st.markdown("""
        <style>
        .main {
            background-color: black;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius:10px;
            border:none;
            padding:10px 24px;
            text-align:center;
            display:inline-block;
            font-size:20px;
            margin:4px 2px;
            transition-duration: 0.4s;
            cursor:pointer;
        }
        .stButton>button:hover {
            background-color: white; 
            color: black; 
            border: 2px solid #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("COVID-19 Detection from X-Ray Images")
    st.write("This application uses deep learning to predict the presence of COVID-19 in X-ray images. Please upload your X-ray image in JPG format.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        img = load_and_prepare_image(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        if st.button('Predict'):
            out_pred, out_prob = predict(img)
            out_prob = out_prob * 100
            st.success('Prediction: {}'.format(out_pred))
            st.info('Probability: {:.2f}%'.format(out_prob))

if __name__ == '__main__':
    main()
