import unittest
from tests.test_api import ApiTest
from tests.test_trainer import TrainerTest

suite = unittest.TestSuite()
suite.addTests(unittest.makeSuite(ApiTest))
suite.addTests(unittest.makeSuite(TrainerTest))

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)
