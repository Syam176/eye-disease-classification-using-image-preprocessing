from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import os
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from config import users_collection, history_collection
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Load the model and labels
model = None
class_names = []

def load_model_and_labels():
    global model, class_names
    try:
        model = load_model("keras_Model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        print("Model and labels loaded successfully")
    except Exception as e:
        print(f"Error loading model or labels: {e}")
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if users_collection.find_one({'email': email}):
            flash('Email already exists')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password
        })
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = users_collection.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out successfully')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        # Ensure user_id exists
        user_id = session.get('user_id')
        if not user_id:
            flash('Session expired. Please login again.')
            return redirect(url_for('login'))
            
        # Get total count with error handling
        try:
            total = history_collection.count_documents({'user_id': user_id})
        except Exception as e:
            print(f"Database count error: {e}")
            total = 0
            
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(max(1, page), total_pages)
        
        # Get paginated results with error handling
        try:
            history_items = list(history_collection.find(
                {'user_id': user_id}
            ).sort('timestamp', -1).skip((page - 1) * per_page).limit(per_page))
        except Exception as e:
            print(f"Database query error: {e}")
            history_items = []
            
        return render_template(
            'history.html',
            history=history_items,
            username=session.get('username'),
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages
        )
        
    except Exception as e:
        print(f"Error in history route: {e}")
        flash('An error occurred while loading history. Please try again later.')
        return render_template(
            'history.html',
            history=[],
            username=session.get('username'),
            page=1,
            per_page=10,
            total=0,
            total_pages=1
        )

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    if 'user_id' not in session:
        print("User not authenticated")
        return jsonify({'error': 'User not authenticated'}), 401

    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        print("Processing image...")
        # Save the file temporarily
        image = Image.open(file.stream)
        
        # Resize the image to match the model's expected input
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Create batch dimension
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])
        
        print(f"Prediction made: {class_name} with confidence {confidence_score}")
        
        # Convert image to base64 for returning in response
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save to history with all required fields
        history_entry = {
            'user_id': session['user_id'],
            'username': session['username'],
            'class': class_name,
            'confidence': confidence_score,
            'image': img_str,
            'timestamp': datetime.now()
        }
        
        print("Attempting to save to history...")
        result = history_collection.insert_one(history_entry)
        print(f"History saved with ID: {result.inserted_id}")
        
        return jsonify({
            'class': class_name,
            'confidence': confidence_score,
            'image': img_str
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model_and_labels()
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
    app.config['SESSION_COOKIE_SECURE'] = True  # For HTTPS
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.run(debug=True)