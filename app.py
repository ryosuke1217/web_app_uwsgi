from flask import Flask, request, jsonify, abort
import json
import logging
import logging.handlers
import validate as vl
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

handler = logging.handlers.RotatingFileHandler("test.log", "a+", maxBytes=3000, backupCount=5)
handler.setLevel(logging.INFO) 
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
app.logger.addHandler(handler)

@app.route("/", methods=['POST'])
def index():
    if not request.method == 'POST':
        abort(405)
    data = json.loads(request.data.decode('utf-8'))
    ret = vl.valid(data)
    if not ret == 'OK':
        app.logger.error('schema error')  
        abort(400)
    return jsonify({"message": "OK"})

@app.errorhandler(400)
def error_handler(error):
    print(error)
    response = jsonify({ 'message': "Invalid parameter", 'result': error.code })
    return response, error.code

@app.errorhandler(405)
def error_handler(error):
    response = jsonify({ 'message': "Method not allowed", 'result': error.code })
    return response, error.code

if __name__ == "__main__":
    app.run(host='192.168.33.10', port=5000, debug=True)