import unittest

from getParamsFromR import *


def checkEqual(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

class TestStringMethods(unittest.TestCase):

    def test_colour2labels(self):

        colour = (  0,  0,  0)
        self.assertIsNone(colour2labels(colour))

        colour = (153,153,153)
        self.assertTrue(checkEqual(colour2labels(colour), ['pole', 'polegroup']))
        colour = (  0,  0,142)
        self.assertTrue(checkEqual(colour2labels(colour), ['car', 'license plate']))

        for name in name2label:
            if name not in ["unlabeled", "ego vehicle", "rectification border", "out of roi", "static", "pole", "polegroup", "car", "license plate"]:
                try:
                    self.assertTrue(checkEqual(colour2labels(name2label[name].color), [name]))
                except AssertionError:
                    print(name + " failed the test.")






if __name__ == '__main__':
    unittest.main()
