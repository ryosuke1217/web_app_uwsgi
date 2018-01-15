from flask import Flask, request, jsonify, abort
from jsonschema import validate, ValidationError
import json
import validate as vl
app = Flask(__name__)

@app.route("/", methods=['POST'])
def index():
    if not request.method == 'POST':
        abort(405)
    data = json.loads(request.data.decode('utf-8'))
    ret = vl.valid(data)
    if not ret == 'OK':
        abort(400)
    return jsonify({"message": "OK"})

@app.errorhandler(400)
def error_handler(error):
    print(error)
    response = jsonify({ 'message': "invalid parameter", 'result': error.code })
    return response, error.code

@app.errorhandler(405)
def error_handler(error):
    response = jsonify({ 'message': "invalid request method", 'result': error.code })
    return response, error.code

if __name__ == "__main__":
    app.run(host='192.168.33.10', port=5000)