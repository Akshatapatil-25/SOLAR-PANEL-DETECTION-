from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
from detect import detect_and_annotate

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define upload and output folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Detection API
@app.route('/detect', methods=['POST'])
def detect():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = "uploaded.jpg"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(upload_path)

    output_filename, detection_data = detect_and_annotate(upload_path)

    return jsonify({
        'output_image': f'/image/{output_filename}',
        'detection_data': detection_data
    })


# Serve detected images
@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(
        app.config['OUTPUT_FOLDER'],
        filename,
        mimetype='image/jpeg'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)