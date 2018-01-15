import unittest
import validate

class TestValidationCheck(unittest.TestCase):

    def test_validation_check(self):
        """正常系テスト
        """
        data = {
            "items": [
                {
                    "item": "coffee", 
                    "sequence": 123
                },
                {
                    "item": "mocha", 
                    "sequence": 456
                }
            ]
        }
        self.assertEqual(validate.valid(data), "OK")

    def test_check_sequence_digit(self):
        """sequence桁数チェック
        """
        data = {
            "items": [
                {
                    "item": "coffee", 
                    "sequence": 12356
                }
            ]
        }
        self.assertNotEqual(validate.valid(data), "OK")
    
    def test_check_items_type(self):
        """items型チェック
        """
        data = {
            "items": {
                "item": "coffee", 
                "sequence": 12356
            }
        }
        self.assertNotEqual(validate.valid(data), "OK")

    def test_check_item_type(self):
        """item型チェック
        """
        data = {
            "items": [
                {
                    "item": True, 
                    "sequence": 123
                }
            ]
        }
        self.assertNotEqual(validate.valid(data), "OK")

    def test_check_required(self):
        """key必須チェック
        """
        data = {
            "items": [
                {
                    "sequence": 123
                }
            ]
        }
        self.assertNotEqual(validate.valid(data), "OK")

        data = {
            "items": [
                {
                    "item": "coffee"
                }
            ]
        }
        self.assertNotEqual(validate.valid(data), "OK")

if __name__ == "__main__":
    unittest.main()