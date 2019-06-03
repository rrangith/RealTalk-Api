from flask import Flask, render_template, jsonify, request, abort
from flask_cors import CORS
import json
from detection import detectImage

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def index():
    return 'hi'

@app.route('/detect', methods=['GET'])
def detect():
    image = request.data.decode("utf-8").split(';base64,')[1][:-1]
    detectImage(image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
