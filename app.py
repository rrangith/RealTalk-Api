from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from detection import detectImage

import base64
import io
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def index():
    return 'hi'

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
    image = Image.open(request.files['file'])
    response = detectImage(image)
    return jsonify(response)

@app.route('/detectLink', methods=['POST'])
def detectLink():
    response = requests.get(request.get_json()['url'])
    image = Image.open(BytesIO(response.content))
    response = detectImage(image)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
