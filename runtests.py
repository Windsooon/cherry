import unittest
from tests.test_trainer import TrainerTest
from tests.test_classify import ClassifyTest

suite = unittest.TestSuite()
suite.addTests(unittest.makeSuite(TrainerTest))
suite.addTests(unittest.makeSuite(ClassifyTest))

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)
