import unittest
from tests.test_trainer import TrainerTest
from tests.test_classify import ClassifyTest
from tests.test_api import ApiTest
from tests.test_display import DisplayTest
from tests.test_search import SearchTest
from tests.test_performance import PerformanceTest

suite = unittest.TestSuite()
suite.addTests(unittest.makeSuite(TrainerTest))
suite.addTests(unittest.makeSuite(ClassifyTest))
suite.addTests(unittest.makeSuite(ApiTest))
suite.addTests(unittest.makeSuite(DisplayTest))

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)
