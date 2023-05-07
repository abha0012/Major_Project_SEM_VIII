'''Stores information about Flask App'''
from flask import Flask
from flask_cors import CORS
UPLOAD_FOLDER = '/Users/aabhaygawande/Desktop/SEM_8_Major_Project/Backend/static/uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024