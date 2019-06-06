from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import base64
import io
from PIL import Image
import requests
from io import BytesIO
import os

from detection import detectImage

app = Flask(__name__)
CORS(app)

def authenticate(key):
    api_key = os.environ.get('API_KEY')
    if key != api_key and api_key != None:
        abort(401)

@app.route('/', methods=['GET'])
def index():
    return 'Hello World'

# @app.route('/detectBinary', methods=['POST'])
# def detectBinary():
#     try:
#         imageBinary = request.data.decode("utf-8").split(';base64,')[1][:-1]
#         img_data = base64.b64decode(str(imageBinary))
#         image = Image.open(BytesIO(img_data))
#         detectImage(image)
#     except:
#         abort(400)

@app.route('/detectFile', methods=['POST'])
def detectFile():
    authenticate(request.headers.get('x-api-key'))
    try:
        image = Image.open(request.files['file'])
        response = detectImage(image)
        return jsonify(response)
    except:
        abort(400)

@app.route('/detectLink', methods=['POST'])
def detectLink():
    authenticate(request.headers.get('x-api-key'))
    try:
        image_data = requests.get(request.get_json()['url'])
        image = Image.open(BytesIO(image_data.content))
        response = detectImage(image)
        return jsonify(response)
    except:
        abort(400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
