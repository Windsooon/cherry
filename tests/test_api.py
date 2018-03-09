import unittest
import cherry


class ApiTest(unittest.TestCase):

    def test_classify(self):
        cherry.classify('警方召开了全省集中打击赌博违法犯罪活动专项行动电视电话会议。会议的重点是“查处”六合彩、赌球赌马等赌博活动。')
