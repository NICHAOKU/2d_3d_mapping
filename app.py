from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import numpy as np
from mapping_algorithm import Image3DMapper
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store current mapper and data
current_mapper = None
current_correspondences = []

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'obj'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global current_mapper, current_correspondences
    
    try:
        print("Upload request received")
        print(f"Request files: {list(request.files.keys())}")
        
        # Check if files are present
        if 'image' not in request.files or 'model' not in request.files:
            print("Missing files in request")
            return jsonify({'error': 'Both image and 3D model files are required'}), 400
        
        image_file = request.files['image']
        model_file = request.files['model']
        
        print(f"Image file: {image_file.filename}, size: {len(image_file.read())} bytes")
        image_file.seek(0)  # Reset file pointer
        print(f"Model file: {model_file.filename}, size: {len(model_file.read())} bytes")
        model_file.seek(0)  # Reset file pointer
        
        # Validate files
        if image_file.filename == '' or model_file.filename == '':
            print("Empty filenames")
            return jsonify({'error': 'No files selected'}), 400
        
        if not (allowed_file(image_file.filename) and allowed_file(model_file.filename)):
            print(f"Invalid file types: {image_file.filename}, {model_file.filename}")
            return jsonify({'error': 'Invalid file types'}), 400
        
        # Save files
        image_filename = secure_filename(image_file.filename)
        model_filename = secure_filename(model_file.filename)
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
        
        print(f"Saving files to: {image_path}, {model_path}")
        
        image_file.save(image_path)
        model_file.save(model_path)
        
        print("Files saved successfully")
        
        # Initialize mapper and process
        print("Initializing Image3DMapper...")
        current_mapper = Image3DMapper(model_path, image_path)
        
        print("Starting processing...")
        current_correspondences = current_mapper.process()
        
        print(f"Processing complete. Correspondences: {len(current_correspondences) if current_correspondences else 0}")
        
        if current_correspondences:
            return jsonify({
                'success': True,
                'message': f'Files processed successfully. Found {len(current_correspondences)} correspondences.',
                'image_path': image_filename,
                'model_path': model_filename,
                'correspondences_count': len(current_correspondences)
            })
        else:
            print("No correspondences found")
            return jsonify({'error': 'Failed to process files - no correspondences found'}), 500
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/map_pixel', methods=['POST'])
def map_pixel():
    global current_mapper
    
    if current_mapper is None:
        return jsonify({'error': 'No files loaded. Please upload files first.'}), 400
    
    try:
        data = request.get_json()
        pixel_x = float(data['x'])
        pixel_y = float(data['y'])
        
        # Map pixel to 3D
        point_3d = current_mapper.map_pixel_to_3d(pixel_x, pixel_y)
        
        if point_3d:
            return jsonify({
                'success': True,
                'pixel': [pixel_x, pixel_y],
                'point_3d': point_3d
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No 3D correspondence found for this pixel'
            })
            
    except Exception as e:
        return jsonify({'error': f'Mapping error: {str(e)}'}), 500

@app.route('/get_correspondences')
def get_correspondences():
    global current_correspondences
    
    return jsonify({
        'correspondences': current_correspondences,
        'count': len(current_correspondences)
    })

@app.route('/get_3d_model')
def get_3d_model():
    global current_mapper
    
    if current_mapper is None:
        print("get_3d_model: current_mapper is None")
        return jsonify({'error': 'No 3D model loaded (mapper is None)'}), 400
    if not hasattr(current_mapper, 'vertices'):
        print("get_3d_model: current_mapper has no 'vertices' attribute")
        return jsonify({'error': 'No 3D model loaded (no vertices attribute)'}), 400
    if current_mapper.vertices is None:
        print("get_3d_model: current_mapper.vertices is None")
        return jsonify({'error': 'No 3D model loaded (vertices is None)'}), 400
    
    print(f"get_3d_model: current_mapper.vertices shape: {current_mapper.vertices.shape}")
    
    try:
        # Convert vertices to JSON format for Three.js
        points = current_mapper.vertices
        faces = None
        
        # Include face data if available
        if hasattr(current_mapper, 'faces') and current_mapper.faces is not None:
            faces = current_mapper.faces.tolist()
            print(f"get_3d_model: current_mapper.faces shape: {current_mapper.faces.shape}")
        
        model_data = {
            'points': points.tolist(),
            'faces': faces,
            'colors': None,  # No colors available from simple OBJ parsing
            'count': len(points),
            'face_count': len(faces) if faces else 0
        }
        
        return jsonify(model_data)
        
    except Exception as e:
        return jsonify({'error': f'Error getting 3D model: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/reset')
def reset():
    global current_mapper, current_correspondences
    current_mapper = None
    current_correspondences = []
    return jsonify({'success': True, 'message': 'Session reset'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)