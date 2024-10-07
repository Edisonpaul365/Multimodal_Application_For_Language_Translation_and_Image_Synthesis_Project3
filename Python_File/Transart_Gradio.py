import os
import requests
from transformers import MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw
import io
import gradio as gr
import torch

# Detect if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MarianMT model and tokenizer for translation (Tamil to English)
model_name = "Helsinki-NLP/opus-mt-mul-en"
translation_model = MarianMTModel.from_pretrained(model_name).to(device)
translation_tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load GPT-Neo for creative text generation
text_generation_model_name = "EleutherAI/gpt-neo-1.3B"
text_generation_model = AutoModelForCausalLM.from_pretrained(text_generation_model_name).to(device)
text_generation_tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name)

# Add padding token to GPT-Neo tokenizer if not present
if text_generation_tokenizer.pad_token is None:
    text_generation_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set your Hugging Face API key
os.environ['HF_API_KEY'] = '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'  # Replace with your actual API key
api_key = os.getenv('HF_API_KEY')
if api_key is None:
    raise ValueError("Hugging Face API key is not set. Please set it in your environment.")

headers = {"Authorization": f"Bearer {api_key}"}

# Define the API URL for image generation (replace with actual model URL)
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"  # Updated model URL

# Query Hugging Face API to generate image with error handling
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response: {response.text}")
        return None
    return response.content

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
        # Return a blank image with error message
        error_img = Image.new('RGB', (300, 300), color=(255, 0, 0))
        d = ImageDraw.Draw(error_img)
        d.text((10, 150), "Image Generation Failed", fill=(255, 255, 255))
        return error_img

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        print(f"Error: {e}")
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

# Custom CSS for styling
css = """
#transart-title {
    font-size: 3em;
    font-weight: bold;
    color: #4A90E2;
    text-align: center;
    margin-bottom: 10px;
}
#transart-subtitle {
    font-size: 1.5em;
    text-align: center;
    color: #7B7B7B;
    margin-bottom: 20px;
}
body {
    background-color: #E5E5E5;
}
.gradio-container {
    font-family: 'Arial', sans-serif;
    border-radius: 20px;
    padding: 30px;
    background: linear-gradient(to right, #FFFFFF, #F0F0F5);
    box-shadow: 0 6px 30px rgba(0, 0, 0, 0.2);
}
.gradio-button {
    background-color: #4A90E2;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: background-color 0.3s ease;
}
.gradio-button:hover {
    background-color: #357ABD;
}
.gradio-textbox {
    border-radius: 10px;
}
"""

# Custom HTML for title and subtitle (can be displayed in Markdown)
title_markdown = """
# <div id="transart-title">A Multimodal Application</div>
### <div id="transart-subtitle">Language Translation and Image Synthesis</div>
"""

# Gradio interface with customized layout and aesthetics
with gr.Blocks(css=css) as interface:
    gr.Markdown(title_markdown)  # Title and subtitle in Markdown
    with gr.Row():
        with gr.Column():
            tamil_input = gr.Textbox(label="Enter Tamil Text", placeholder="Type Tamil text here...", lines=3, interactive=True)  # Input for Tamil text
        with gr.Column():
            translated_output = gr.Textbox(label="Translated Text", interactive=False)        # Output for translated text
            creative_text_output = gr.Textbox(label="Creative Generated Text", interactive=False)  # Output for creative text
            generated_image_output = gr.Image(label="Generated Image")  # Output for generated image

    gr.Button("Generate").click(fn=translate_generate_image_and_text, inputs=tamil_input, outputs=[translated_output, creative_text_output, generated_image_output])

# Launch the Gradio app
interface.launch(debug=True, server_name="0.0.0.0", share=True)
