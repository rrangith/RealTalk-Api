from flask import Flask, render_template, jsonify, request, abort
from flask_cors import CORS
import json
from detection import detectImage

import base64
import io
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def index():
    return 'hi'

@app.route('/detectBinary', methods=['POST'])
def detectBinary():
    # try:
    imageBinary = request.data.decode("utf-8").split(';base64,')[1][:-1]
    img_data = base64.b64decode(str(imageBinary))
    image = Image.open(io.BytesIO(img_data))
    detectImage(image)
    # except:
    #     abort(400)

@app.route('/detectFile', methods=['POST'])
def detectFile():
    image = Image.open(request.files['file'])
    detectImage(image)
    return 'hi'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
