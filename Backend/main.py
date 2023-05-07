# pylint: disable=C0301,C0103
'''Trial code with improved pylint score --> Main code in main.py'''
import os
import pathlib
import nibabel as nib
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.image
from flask_cors import cross_origin, CORS


from tensorflow import keras
from werkzeug.utils import secure_filename
from flask import render_template, flash, request
from keras.utils import img_to_array, load_img
from scipy import ndimage
from matplotlib import pyplot, transforms
from app import app

ALLOWED_EXTENSIONS = set(['nii'])

A1_model = tf.keras.models.load_model('/Users/aabhaygawande/Desktop/SEM_8_Major_Project/Backend/A1_Model')

def allowed_file(filename):
    '''To check if the uploaded file extension is in the list'''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


@app.route('/')
def upload_form():
    '''Route to original landing/home page'''
    return render_template('upload.html')

@app.route('/mri', methods=['POST'])
def upload_image():
    '''Route when image is uploaded, and user is expecting for result'''
    
    if 'file' not in request.files:
        flash('No file part')
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return "No image selected for uploading"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = os.path.join("/Users/aabhaygawande/Desktop/SEM_8_Major_Project/Backend/static/uploads", filename)
        # Read and process the scans.
        # Each scan is resized across height, width, and depth and rescaled.
        scan = process_scan(path)
        scan = ndimage.rotate(scan, -90, reshape=False)
        img_path = '/Users/aabhaygawande/Desktop/SEM_8_Major_Project/Backend/static/uploads/temp.png'
        matplotlib.image.imsave(img_path, np.squeeze(scan[:, :, 30]))
        volume = rotate(scan)
        volume = tf.expand_dims(volume, axis=3)
        classes = ['Mild Cognitive Impairment','Alzheimers Disease', 'Cognitively Normal']
        temp = A1_model.predict(np.array([volume]))
        print('Class and image_path',classes[np.argmax(temp[0])],img_path)
    response = classes[np.argmax(temp[0])] + '^' +  img_path
    return {"output": response}

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

if __name__ == "__main__":
    CORS(app, supports_credentials=True)
    cors = CORS(app, resource={
        r"/*":{
            "origins":"*"
        }
    })
    app.run()
