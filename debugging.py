import os # Interact with operating system
import tensorflow as tf # Build and train models
import tensorflow_hub as hub # Load pre-trained models from TensorFlow hub
import cv2 # OpenCV library for image processing (computer vision)
import pytesseract # OCR library for extracting text from images
from flask import Flask, request, jsonify # Flask framework for building web server
from flask_cors import CORS # CORS for cross-origin requests
from flask_limiter import Limiter # Flask extension for rate limiting
from flask_limiter.util import get_remote_address # Utility function --> Rate limiting based on IP address
from redis import Redis
import base64 # Encode and decode base64 strings
import io # Provide Python interfaces to stream handling
from PIL import Image # Python Imaging Library (PIL) for image manipulation
import numpy as np # Numerical operations on arrays
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # HuggingFace library to access LLaMA-3.1-8B-instruct
import logging # Standard Python logging library
from werkzeug.middleware.proxy_fix import ProxyFix # Proxy middleware to ensure Flask can correctly handle request behind a reverse proxy
from flask_sqlalchemy import SQLAlchemy # Flask extension for database interaction
from flask_bcrypt import Bcrypt # Bcrypt for hashing passwords
from flask_jwt_extended import  JWTManager, create_access_token, jwt_required, get_jwt_identity # JWT for authenticationimport jwt # Create tokens for user authentication
# import datetime # Handle date and time opertions
from werkzeug.security import generate_password_hash, check_password_hash # Secure user password
# import json
import requests
import torch 
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Initialize Flask app
app = Flask(__name__)
CORS(app)
redis_client = Redis(host='localhost', port=8000, db=0)

# Apply ProxyFix to handle reverse proxy setup
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# Configure database to store user information
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db' # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key' # encode tokens


# initialize database
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Define a user model (table in database)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) # unique identifier for each user
    username = db.Column(db.String(150), unique=True, nullable=False) # username --> must be unique
    password = db.Column(db.String(150), nullable=False) # password --> hashed for security

# Create database and database table
with app.app_context():
    db.create_all()

# Initialize rate limiter (create web server and apply rate limiting)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri = "redis://localhost:8000/0",
    default_limits=["100 per day", "20 per hour"] # limits to support up to 500 users per day
)
limiter.init_app(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Check if input is a base64-encoded image
def is_base64_image(input_data):
    try:
        image_data = base64.b64decode(input_data)
        image = Image.open(io.BytesIO(image_data))
        return True
    except Exception:
        return False
    
# Check if image is a code snippet
def is_code_image(image_path):
    # Convert image to grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OCR to extract text
    text = pytesseract.image_to_string(gray_image)

    # Check for common code text patterns
    code_keywords = ["def", "class", "import", "console.log", "<html>", "function", "return", 
        "var", "let", "const", "public", "private", "protected", "void", "int", 
        "if", "else", "elif", "while", "for", "foreach", "switch", "case"]
    
    return any(keyword in text for keyword in code_keywords)

# Check if image is a frontend screenshot
def is_frontend_image(image_path):
    try:
        # Convert image to grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find UI components
        edges = cv2.Canny(gray_image, 50, 150)

        # Use contours to find rectangular shapes
        contours, _ =  cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4: # rectangular shape
                return True
        return False
    
    except Exception as e:
        logging.error(f"Error determining frontend image: {e}")
        return False
    
# Check if image is a terminal screenshot
def is_terminal_image(image_path):
    try:
        # Convert image to grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use OCR to extract text
        text = pytesseract.image_to_string(gray_image)

        # Check for common terminal text patterns
        terminal_keywords = ["$", "%", ">>>", "command not found", "error", 
            "Permission denied", "Traceback"]

        return any(keyword in text for keyword in terminal_keywords)
    
    except Exception as e:
        logging.error(f"Error determining terminal image: {e}")
        return False

# Handle images based on type
def handle_image(input_data):
    # Decode the base64 image data
    image_data = base64.b64decode(input_data)
    image = Image.open(io.BytesIO(image_data))
    image_path = "temp_image.png"
    image.save(image_path)

    # Determine specific type of image
    if is_code_image(image_path):
        response = debug_code_from_image(image_path)
    elif is_terminal_image(image_path):
        response = analyze_terminal_screenshot(image_path)
    elif is_frontend_image(image_path):
        response = generate_frontend_code_from_image(image_path)
    else:
        response = "Unknown image type. Please provide a description."

    # Remove temporary image file
    os.remove(image_path)
    return response



# Load pre-trained TensorFlow models for image classification
image_captioning_model_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/5" # PLACEHOLDER URL for image captioning model (translating frontend ss to code) --> REVIEW
try:
    image_captioning_model = hub.KerasLayer(image_captioning_model_url)
except Exception as e:
    logging.error(f"Failed to load image captioning model: {e}")
    raise RuntimeError(f"Failed to load image captioning model: {e}")

# Load pre-trained LLaMA model for text generation (debugging)
llama_model_name = "meta-llama/Meta-Llama-3.1-8B"
try:
    # Load model directly from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token to EOS token
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, device_map="mps", torch_dtype=torch.float16, offload_folder="offload_folder") # Load model in 4-bit precision

    # Move model to CPU device
    device = torch.device("mps") # Handle device compatibility check (mc\]]
    logging.info(f"Using device: {device}")
except Exception as e:
    logging.error(f"Failed to load LLaMA-3.1 model: {e}")
    raise RuntimeError(f"Failed to load LLaMA-3.1 model: {e}")





# Define the OCR function
def extract_text_from_image(image_path):
    try: 
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_image)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return f"Error extracting text from image: {e}"

# Define functions for handling different screenshots
def generate_frontend_code_from_image(image_path):
    try:
        image = Image.open(image_path)
        image = np.array(image) / 255.0  # Normalize image
        image = tf.image.resize(image, [224, 224])
        image = np.expand_dims(image, axis=0)

        # Generate caption using the image captioning model
        predictions = image_captioning_model(image)
        caption_text = tf.nn.softmax(predictions[0])
        predicted_class = np.argmax(caption_text)

        # Converting caption to frontend code 
        labels_path = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        class_labels = requests.get(labels_path).json()
        caption_text = class_labels[str(predicted_class)][1]
        code_text = generate_response(f"Write code to design a frontend based on the following description: {caption_text}")
        return f"Generated frontend code:\n{code_text}"
    
    except Exception as e:
        logging.error(f"Error generating frontend code from image: {e}")
        return f"Error generating frontend code from image: {e}"

# Debug code with memory constraints
def debug_code_with_model(code_text):
    try:
        logging.info("Starting to debug code with the model...")

        # Tokenize input text
        prompt = f"Below is a snippet of code that contains errors. Please correct the errors and provide the corrected code.\n{code_text}"
        logging.info(f"Generated prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs.input_ids.to(device, dtype=torch.long)
        attention_mask = inputs.attention_mask.to(device, dtype=torch.long)
        logging.info(f"Tokenized inputs: {inputs}")

        # Generate output using LLaMA-3.1-8B model
        with torch.no_grad(): # attention_mask=attention_mask
            outputs = llama_model.generate(input_ids, attention_mask=attention_mask, max_length=500)
        logging.info(f"Model output generated.")
        debugged_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Generated text: {debugged_code}")
        return debugged_code
    
    except Exception as e:
        logging.error(f"Error debugging code with model: {e}")
        return f"Error debugging code with model: {e}"

def debug_code_from_image(image_path):
    try:
        # Extract text from image using the defined OCR function
        code_text = extract_text_from_image(image_path)

        # Debug the code using the LLaMA-3.1-8B model
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
        
        # Use LLaMA model to generate response based on extracted text
        response = generate_response(f"Please analyze the following terminal and find a solution:\n{text}")
        response = f"Solution: {response}"
        return response
    
    except Exception as e:
        logging.error(f"Error analyzing terminal screenshot: {e}")
        return f"Error analyzing terminal screenshot: {e}"

# Define the function to generate a response using the LLaMA model for general prompts
def generate_response(prompt):
    try:
        # Tokenize input text
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

        # Generate output using LLaMA-3.1-8B model
        outputs = llama_model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_length=300, num_return_sequences=1)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Ensure prompt is not included in response
        response_text = response_text.replace(prompt, "").strip()
      
        # Return the generated response
        response = f"Prompt: {prompt}\nGenerated response: {response_text}"
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error generating response: {e}"

# Route for user registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "Username and password are required"}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password=hashed_password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"messge": "User registered successfully"}), 201
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return jsonify({"message": "Error registering user"}), 500


# Route for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "Username and password are required"}), 400
    
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        access_token = create_access_token(identity=user.id)
        return jsonify(access_token=access_token)
    else:
        return jsonify({"message": "Invalid credentials"}), 401

# Define and secure API endpoints
@app.route('/process', methods=['POST'])
@jwt_required() # ensure only authenticated users can access this endpoint
@limiter.limit("10 per minute")
def process_input():
    data = request.json
    input_data = data.get('input_data')

    if not input_data:
        return jsonify({"response": "Missing input data"}), 400

    # Process user input based on type
    try: 
        if is_base64_image(input_data):
            response = handle_image(input_data)
        elif isinstance(input_data, str):
            response = generate_response(input_data)
        else:
            response = "Invalid input type."

        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return jsonify({"response": str(e)}), 500
    
def main():
    # # Sample inputs
    # sample_text1 = "What is the difference between a stack and a queue?"
    # sample_text2 = "How do I make a list in Python?"
    # with open("test_image.png", "rb") as sample_image_file:
    #     base64_image = base64.b64encode(sample_image_file.read()).decode('utf-8')
    #     print(handle_image(base64_image))

    # # Test generate_response()
    # print(generate_response(sample_text1))
    # print(generate_response(sample_text2))
    code_text = 'print("Hello World!\nprimt("My name is David")\nprint("I like swimming and cats"\nMy favorite cat is called Whiskers'
    print("Over here!")
    print(debug_code_with_model(code_text))
    # print(generate_response(prompt1))
    # print(generate_response(prompt2))
        



# Run the Flask app with Gunicorn for production
if __name__ == '__main__':
    main()
