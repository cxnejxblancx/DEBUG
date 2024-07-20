import os
import tensorflow as tf 
import tensorflow_hub as hub
import cv2
import pytesseract
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import base64 
import io
from PIL import Image
import numpy as np
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel 
import logging


# Initialize Flask app
app = Flask(__name__)

# Initialize rate limiter (create web server and apply rate limiting)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per day", "20 per hour"] # limits to support up to 500 users per day
)
limiter.init_app(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained TensorFlow models (frontend)
image_captioning_model_url = "https://tfhub.dev/tensorflow/tf2-preview/generator/1"
try:
    image_captioning_model = hub.load(image_captioning_model_url)
except Exception as e:
    logging.error(f"Failed to load image captioning model: {e}")
    raise RuntimeError(f"Failed to load image captioning model: {e}")

# Load pre-trained GPT-2 model for text generation (debugging)
gpt2_model_name = "gpt2"
try:
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = TFGPT2LMHeadModel.from_pretrained(gpt2_model_name)
except Exception as e:
    logging.error(f"Failed to load GPT-2 model: {e}")
    raise RuntimeError(f"Failed to load GPT-2 model: {e}")

# Define the OCR function
def extract_text_from_image(image_path):
    try: 
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_image)
        return text
    except Exception as e:
        logging.error(f"Eror extracting text from image: {e}")
        return f"Error extracting text from image: {e}"

# Define functions for handling different screenshots
def generate_frontend_code_from_image(image_path):
    try:
        image = Image.open(image_path)
        image = np.array(image) / 255.0  # Normalize image
        image = tf.image.resize(image, [224, 224])
        image = np.expand_dims(image, axis=0)

        # Generate caption using the image captioning model
        captions = image_captioning_model(image)
        caption_text = captions.numpy()[0].decode('utf-8')

        # Dummy implementation of converting caption to frontend code  --> REVIEW
        code_text = "Generated frontend code snippet based on the image: " + caption_text
        return code_text
    except Exception as e:
        logging.error(f"Error generating frontend code from image: {e}")
        return f"Error generating frontend code from image: {e}"

def debug_code_with_model(code_text):
    try:
        # Tokenize input text
        inputs = tokenizer.encode(code_text, return_tensors='tf')

        # Generate output using GPT-2 model
        outputs = gpt2_model.generate(inputs, max_length=500, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the debugged code
        debugged_code = "Debugged Code:\n" + generated_text
        return debugged_code
    except Exception as e:
        logging.error(f"Error debugging code with model: {e}")
        return f"Error debugging code with model: {e}"

def debug_code_from_image(image_path):
    try:
        # Extract text from image using the defined OCR function
        code_text = extract_text_from_image(image_path)

        # Debug the code using the GPT-2 model
        debugged_code = debug_code_with_model(code_text)

        return debugged_code
    except Exception as e:
        logging.error(f"Error debugging code from image: {e}")
        return f"Error debugging code from image: {e}"

def analyze_terminal_screenshot(image_path):
    try:
        # Extract text from image using OCR
        text = extract_text_from_image(image_path)
        if "Error" in text:
            return text
        
    # Use GPT-2 model to generate response based on extracted text
        response = f"Solution: {generate_response(text)}"
        return response
    
    except Exception as e:
        logging.error(f"Error analyzing terminal screenshot: {e}")
        return f"Error analyzing terminal screenshot: {e}"

# Define the function to generate a response using the GPT-2 model for general prompts
def generate_response(prompt):
    try:
        # Tokenize input text
        inputs = tokenizer.encode(prompt, return_tensors='tf')

        # Generate output using GPT-2 model
        outputs = gpt2_model.generate(inputs, max_length=500, num_return_sequences=1)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the generated response
        response = f"Generated response for the prompt: {response_text}"
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error generating resonse: {e}"

# Define API endpoints
@app.route('/process', methods=['POST'])
@limiter.limit("10 per minute")
def process_input():
    data = request.json
    input_type = data.get('input_type')
    input_data = data.get('input_data')

    if not input_type or not input_data:
        return jsonify({"response": "Missing input_type or input_data"}), 400

    try: # make sure images properly labeled
        if input_type in ["code_screenshot", "frontend_screenshot", "terminal_screenshot"]:
            image_data = base64.b64decode(input_data)
            image = Image.open(io.BytesIO(image_data))
            image_path = "temp_image.png"
            image.save(image_path)

            if input_type == "frontend_screenshot":
                response = generate_frontend_code_from_image(image_path)
            elif input_type == "code_screenshot":
                response = debug_code_from_image(image_path)
            elif input_type == "terminal_screenshot":
                response = analyze_terminal_screenshot(image_path)

            # Remove the temporary image file
            os.remove(image_path)
        elif input_type == "prompt":
            response = generate_response(input_data)
        else:
            response = "Invalid input type."

        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return jsonify({"response": str(e)}), 500
