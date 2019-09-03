import os
import unittest
from unittest import mock
from cherry.base import DATA_DIR, load_data

class SearchTest(unittest.TestCase):

    def setUp(self):
        self.x_data = [1, 2]
        self.y_data = [0, 1]
