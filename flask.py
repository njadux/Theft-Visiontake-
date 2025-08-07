import os
import logging
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "theft-detection-secret-key")

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Device configuration
device = torch.device('cpu')

# Initialize model variables
model = None
transform = None

def initialize_model():
    """Initialize the PyTorch model"""
    global model, transform
    try:
        # Load model
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        model_path = os.path.join(MODEL_FOLDER, "resnet_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            app.logger.info("Model loaded successfully")
        else:
            app.logger.warning(f"Model file not found at {model_path}")
            model = None
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    except Exception as e:
        app.logger.error(f"Error initializing model: {str(e)}")
        model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, num_frames=16):
    """Extract frames from video for analysis"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            app.logger.error(f"Could not open video file: {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            app.logger.error("Video has no frames")
            cap.release()
            return None
            
        step = max(total_frames // num_frames, 1)
        frames = []
        
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if transform is not None:
                frame = transform(frame)
            frames.append(frame)
            
        cap.release()
        
        if len(frames) == 0:
            return None
        return torch.stack(frames)
        
    except Exception as e:
        app.logger.error(f"Error extracting frames: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Check API and model status"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'online',
        'model_status': model_status,
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

@app.route('/api/test', methods=['POST'])
def test_upload():
    """Simple test endpoint for file uploads"""
    try:
        app.logger.info("Test upload endpoint called")
        
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            })
            
        file = request.files['video']
        app.logger.info(f"File received: {file.filename}, size: {file.content_length}")
        
        return jsonify({
            'success': True,
            'message': 'File upload test successful',
            'filename': file.filename
        })
        
    except Exception as e:
        app.logger.error(f"Test upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for video analysis"""
    filepath = None
    try:
        app.logger.info("Predict endpoint called")
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure the model file exists in the model directory.'
            }), 500

        # Check if file is present - simplified check
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400

        file = request.files['video']
        
        # Check if file is selected
        if not file.filename:
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file format. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Save file
        filename = secure_filename(file.filename or 'video')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        app.logger.info(f"Processing video: {filename}")

        # Extract frames
        frames = extract_frames(filepath)
        if frames is None:
            # Clean up file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'error': 'Could not extract valid frames from the video. Please check the file format and try again.'
            }), 400

        # Make predictions
        predictions = []
        with torch.no_grad():
            for frame in frames:
                frame = frame.unsqueeze(0).to(device)
                output = model(frame)
                pred = torch.softmax(output, dim=1)
                predictions.append(pred.cpu())

        # Calculate average prediction
        avg_pred = torch.mean(torch.stack(predictions), dim=0)
        label = torch.argmax(avg_pred).item()
        confidence = torch.max(avg_pred).item()

        class_names = ['normal', 'theft']
        prediction = class_names[label]

        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

        app.logger.info(f"Prediction complete: {prediction} ({confidence:.4f})")

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'frames_analyzed': len(frames)
        })

    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        # Clean up file if it exists
        try:
            if 'filepath' in locals() and filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        return jsonify({
            'success': False,
            'error': f'An error occurred during processing: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size allowed: {MAX_FILE_SIZE // (1024*1024)}MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    app.logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred'
    }), 500

# Initialize model on startup
initialize_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
