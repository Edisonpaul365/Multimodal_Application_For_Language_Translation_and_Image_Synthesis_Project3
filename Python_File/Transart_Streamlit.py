import streamlit as st
import base64
import os
import requests
from transformers import MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw
import io
import torch

# Detect if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MarianMT model and tokenizer for translation (Tamil to English)
model_name = "Helsinki-NLP/opus-mt-mul-en"
translation_model = MarianMTModel.from_pretrained(model_name).to(device)
translation_tokenizer = MarianTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Set padding token for the translation tokenizer
translation_tokenizer.pad_token = translation_tokenizer.eos_token

# Load GPT-Neo for creative text generation
text_generation_model_name = "EleutherAI/gpt-neo-1.3B"
text_generation_model = AutoModelForCausalLM.from_pretrained(text_generation_model_name).to(device)
text_generation_tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name, clean_up_tokenization_spaces=True)

# Set padding token for the text generation tokenizer
text_generation_tokenizer.pad_token = text_generation_tokenizer.eos_token

# Set your Hugging Face API key
os.environ['HF_API_KEY'] = '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'  # Replace with your actual API key
api_key = os.getenv('HF_API_KEY')
if api_key is None:
    raise ValueError("Hugging Face API key is not set. Please set it in your environment.")

headers = {"Authorization": f"Bearer {api_key}"}

# Define the API URL for image generation
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"

# Function to set background image
def set_background_image(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded image file and encode it in base64
        image_bytes = uploaded_file.read()
        b64 = base64.b64encode(image_bytes).decode()
        # Set the CSS style to set the background image
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{b64});
                background-size: cover;
                background-position: center;
                height: 100vh;
                width: 100vw;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Query Hugging Face API to generate image with error handling
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
        return None

# Translate Tamil text to English
def translate_text(tamil_text):
    inputs = translation_tokenizer(tamil_text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = translation_model.generate(**inputs)
    translation = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation

# Generate an image based on the translated text with error handling
def generate_image(prompt):
    image_bytes = query({"inputs": prompt})

    if image_bytes is None:
        # Return a blank image with an error message
        error_img = Image.new('RGB', (300, 300), color=(255, 0, 0))
        d = ImageDraw.Draw(error_img)
        d.text((10, 150), "Image Generation Failed", fill=(255, 255, 255))
        return error_img

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        # Return an error image in case of failure
        error_img = Image.new('RGB', (300, 300), color=(255, 0, 0))
        d = ImageDraw.Draw(error_img)
        d.text((10, 150), "Invalid Image Data", fill=(255, 255, 255))
        return error_img

# Generate creative text based on the translated English text
def generate_creative_text(translated_text):
    inputs = text_generation_tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True).to(device)
    generated_tokens = text_generation_model.generate(**inputs, max_length=100)
    creative_text = text_generation_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return creative_text

# Function to handle the full workflow
def translate_generate_image_and_text(tamil_text):
    # Step 1: Translate Tamil to English
    translated_text = translate_text(tamil_text)

    # Step 2: Generate an image from the translated text
    image = generate_image(translated_text)

    # Step 3: Generate creative text from the translated text
    creative_text = generate_creative_text(translated_text)

    return translated_text, creative_text, image

# Streamlit app title and subtitle
st.markdown("<h1 style='text-align: center;'>A Multimodal Application</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Language Translation and Image Synthesis</h3>", unsafe_allow_html=True)

# Upload image for background
uploaded_file = st.file_uploader("Upload a JPEG image for the background", type=["jpg", "jpeg"])

# Set the background image if uploaded
set_background_image(uploaded_file)

# User input for Tamil text
tamil_input = st.text_area("Enter Tamil Text", height=150)

# Button to generate translation, creative text, and image
if st.button("Generate"):
    if tamil_input:
        with st.spinner("Generating..."):
            translated_output, creative_text_output, generated_image_output = translate_generate_image_and_text(tamil_input)
        
        st.subheader("Translated Text")
        st.write(translated_output)
        
        st.subheader("Creative Generated Text")
        st.write(creative_text_output)
        
        st.subheader("Generated Image")
        st.image(generated_image_output, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter some Tamil text.")

# Run the Streamlit app
if __name__ == "__main__":
    st.title("Multimodal Application")
