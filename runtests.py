import unittest
from tests.test_api import ApiTest
from tests.test_trainer import TrainerTest
from tests.test_error import ErrorTest

suite = unittest.TestSuite()
suite.addTests(unittest.makeSuite(ApiTest))
suite.addTests(unittest.makeSuite(TrainerTest))
suite.addTests(unittest.makeSuite(ErrorTest))

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)
