from flask import Flask
import json
from jsonschema import validate
app = Flask(__name__)

schema = {
    'type': 'object',
    'properties': {
        'items': {
            'item': {
                'type': 'string'
            },
            'sequence': {
                'type': 'int'
            }
            'type': 'array',
        }
    },
    'required': [
        'items',
        'item',
        'name'
    ]
}

@app.route("/")
def hello():
    return "Hello Flask!"

if __name__ == "__main__":
    app.run()