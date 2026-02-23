import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app

class TestJobCheck(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'JobCheck', response.data)

    def test_exploration_page(self):
        response = self.app.get('/exploration')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Data Exploration', response.data)

    def test_nlp_analysis_page(self):
        response = self.app.get('/nlp_analysis')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'NLP Analysis', response.data)

    def test_model_training_page(self):
        response = self.app.get('/model_training')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Model Training', response.data)

    def test_prediction_page(self):
        response = self.app.get('/prediction')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Real-Time Prediction', response.data)

    def test_invalid_route(self):
        response = self.app.get('/invalid')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()