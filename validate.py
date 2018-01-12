
from jsonschema import validate, ValidationError
import json

# schema = {
#     "type": "object",
#     "properties": {
#         "items": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "item": {
#                         "type": "string",
#                         "minLength": 1
#                     },
#                     "sequence": {
#                         "type": "int",
#                         "minimum": 0,
#                         "maximum": 999
#                     }
#                 },
#                 "required": [
#                     "item",
#                     "sequence"
#                 ]
#             }
#         },
#         "required": ["items"]
#     }
# }

def valid(data):
    if "items" not in data.keys():
        return "キーが不正です"
    return checkObj(data["items"])

def checkObj(data):
    if not isinstance(data, list):
        return "itemの値が配列ではありません。"
    
    for obj in data:
        print(obj)
        if not obj.keys() >= {"item", "sequence"}:
            return "キーが不正です。"
        if not checkItem(obj["item"]) or not checkSequence(obj["sequence"]):
            return "値が不正です。"
    return "OK"

def checkItem(val):
    if not isinstance(val, str):
        return False
    if not len(val) >= 1:
        return False
    return True

def checkSequence(val):
    if not isinstance(val, int):
        return False
    if not not val >= 0 and not val <= 999:
        return False
    return True