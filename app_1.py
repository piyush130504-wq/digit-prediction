import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

st.markdown(
    """
    <style>
    /* Change background color */
    .main {
        background-color: #1e1e2f;  /* Dark blue-ish */
        color: #e0e0e0;  /* Light text color */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Center the canvas and the button */
    .canvas-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Customize the title font */
    .css-18e3th9 h1 {
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        color: #ffb347;  /* Orange-ish */
    }

    /* Customize subheaders */
    .css-1v0mbdj h2 {
        font-family: 'Arial Black', Gadget, sans-serif;
        color: #ffd700; /* Gold */
    }

    /* Customize normal text */
    .css-1d391kg p {
        font-size: 18px;
        font-family: 'Verdana', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Handwritten Digit Recognition (0-9)")

st.write("Draw a digit (0-9) below and click 'Predict Digit' to see the model's prediction.")


st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'), mode='L')
        img = img.resize((28, 28))
        img_arr = np.array(img)
        img_arr = img_arr / 255.0  # Normalize
        img_arr = img_arr.reshape(1, 28, 28, 1)

        model = tf.keras.models.load_model("mnist_strong_cnn1.h5")
        prediction = model.predict(img_arr)
        pred_digit = np.argmax(prediction)

        st.subheader(f"Predicted Digit: {pred_digit}")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit first.")

st.markdown("---")
st.write("Made by Sarvagya Dwivedi")
