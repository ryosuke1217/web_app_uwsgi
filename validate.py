
from jsonschema import validate, ValidationError
import cerberus
import json

input_schema = {
    "items": {
        "required": True,
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "item": {
                    "required": True,
                    "type": "string",
                    "minlength": 1,
                    "empty": False
                },
                "sequence": {
                    "required": True,
                    "type": "integer",
                    "min": 0,
                    "max": 999,
                    "empty": False
                }
            }
        }
    }
}

output_schema = {
    "results": {
        "required": True,
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "sequence": {
                    "required": True,
                    "type": "integer",
                    "min": 0,
                    "max": 9999999999,
                    "empty": False
                 },
                "status": {
                    "required": True,
                    "type": "integer",
                    "min": 0,
                    "max": 1,
                    "empty": False
                },
                "input_cnt": {
                    "type": "integer",
                    "min": 0,
                    "max": 999
                },
                "unk_cnt": {
                    "type": "integer",
                    "min": 0,
                    "max": 999
                },
                "rec_top10": {
                    "type": "list",
                    "schema": {
                        "type": "dict",
                        "schema": {
                            "cat_id": {
                                "type": "integer",
                                "min": 0
                            },
                            "likelihood": {
                                "type": "float",
                                "min": 0,
                                "max": 9999999999
                            }
                        }
                    }
                }
            }
        }
    }
}

def valid(data):
    v = cerberus.Validator(input_schema)
    print(v.validate(data))
    print(v.errors)
    if not v.validate(data):
        return json.dumps(v.errors)
    return "OK"

# def valid(data):
#     if "items" not in data.keys():
#         return "itemsが存在しません"
#     return checkObj(data["items"])

# def checkObj(data):
#     if not isinstance(data, list):
#         return "itemsの値が配列ではありません。"
    
#     for obj in data:
#         print(obj)
#         if not obj.keys() >= {"item", "sequence"}:
#             return "キーが不正です。"
#         if not checkItem(obj["item"]) or not checkSequence(obj["sequence"]):
#             return "値が不正です。"
#     return "OK"

# def checkItem(val):
#     if not isinstance(val, str):
#         return False
#     if not len(val) >= 1:
#         return False
#     return True

# def checkSequence(val):
#     if not isinstance(val, int):
#         return False
#     if not not val >= 0 and not val <= 999:
#         return False
#     return True