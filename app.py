import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json

# Load the feature extraction model
feature_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load the trained caption generation model
caption_model = load_model('model.h5')

# Load the tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Define the maximum length of the captions (from your training data)
max_len = 30 

def extract_features(image, model):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

def generate_caption(image, feature_model, caption_model, tokenizer, max_length):
    # Extract features
    features = extract_features(image, feature_model)

    # Generate caption
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Streamlit application starts here
def main():
    st.title("Image Captioning App")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        caption = generate_caption(image, feature_model, caption_model, tokenizer, max_len)
        st.write("Generated Caption:", caption)

if __name__ == "__main__":
    main()

