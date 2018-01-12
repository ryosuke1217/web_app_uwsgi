from flask import Flask, request
from jsonschema import validate, ValidationError
import json
import validate as vl
app = Flask(__name__)

@app.route("/", methods=['POST'])
def index():
    data = json.loads(request.data.decode('utf-8'))
    ret = vl.valid(data)
    return ret
    # print(data)
    # # print(schema)
    # try:
    #     validate(data, schema)
    # except ValidationError as e:
    #     print('Invalid JSON - {0}'.format(e.message))
    # return data["items"][0]["item"]

if __name__ == "__main__":
    app.run(host='192.168.33.10', port=5000)