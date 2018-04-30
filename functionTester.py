import unittest

from getParamsFromR import *
from virtObject import *


exhaustive = True

def checkEqualList(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

class TestInputAnalyser(unittest.TestCase):

    def test_colour2labels(self):

        colour = (  0,  0,  0)
        self.assertIsNone(colour2labels(colour))

        colour = (153,153,153)
        self.assertTrue(checkEqualList(colour2labels(colour), ['pole', 'polegroup']))
        colour = (  0,  0,142)
        self.assertTrue(checkEqualList(colour2labels(colour), ['car', 'license plate']))

        for name in name2label:
            if name not in ["unlabeled", "ego vehicle", "rectification border", "out of roi", "static", "pole", "polegroup", "car", "license plate"]:
                try:
                    self.assertTrue(checkEqualList(colour2labels(name2label[name].color), [name]))
                except AssertionError:
                    print(name + " failed the test.")




class TestVirtObject(unittest.TestCase):

    def test_pointInPrism(self):
        eightpoints = [[1,1,1], [1,3,1], [3,3,1], [3,1,1],
                    [1,1,3], [1,3,3], [3,3,3], [3,1,3]]

        rprism = RectPrism(eightpoints)

        self.assertFalse(rprism.contains((0,0,0)))
        self.assertFalse(rprism.contains((0.5,0.5,0.5)))
        self.assertFalse(rprism.contains((3,3,3)))

        self.assertTrue(rprism.contains((1.5,1.5,1.5)))
        self.assertTrue(rprism.contains((2,2,2)))
        self.assertTrue(rprism.contains((2.5,2.5,2.5)))


    def test_wrongOrder_pointInPrism(self):

        # eightpoints = [[1,1,1], [1,3,1], [3,3,1], [3,1,1], [1,1,3], [1,3,3], [3,3,3], [3,1,3]]
        eightpoints = [[1,3,3], [3,3,3], [3,1,3], [1,1,1], [1,3,1], [3,3,1], [3,1,1], [1,1,3]]

        rprism = RectPrism(eightpoints)

        self.assertFalse(rprism.contains((0,0,0)))
        self.assertFalse(rprism.contains((0.5,0.5,0.5)))
        self.assertFalse(rprism.contains((3,3,3)))

        self.assertTrue(rprism.contains((1.5,1.5,1.5)))
        self.assertTrue(rprism.contains((2,2,2)))
        self.assertTrue(rprism.contains((2.5,2.5,2.5)))


    @unittest.skipIf(exhaustive == False, "Exhaustive tests are deactivated")
    def test_extensive_pointInPrism(self):

        eightpoints = [[1,1,1], [1,3,1], [3,3,1], [3,1,1],
                    [1,1,3], [1,3,3], [3,3,3], [3,1,3]]

        rprism1 = RectPrism(eightpoints)

        for i in range(0, 50):
            for j in range(0, 50):
                for k in range(0, 50):
                    x = i/10.0
                    y = j/10.0
                    z = k/10.0
                    if (1.0 < x < 3.0) and (1.0 < y < 3.0) and (1.0 < z < 3.0):
                        self.assertTrue(rprism1.contains((x,y,z)))
                    else:
                        self.assertFalse(rprism1.contains((x,y,z)))

    @unittest.skipIf(exhaustive == False, "Exhaustive tests are deactivated")
    def test_extensive_neg_pointInPrism(self):

        eightpoints = [[-1,-1,-1], [-1,-3,-1], [-3,-3,-1], [-3,-1,-1],
                    [-1,-1,-3], [-1,-3,-3], [-3,-3,-3], [-3,-1,-3]]

        rprism1 = RectPrism(eightpoints)

        for i in range(0, -50):
            for j in range(0, -50):
                for k in range(0, -50):
                    x = i/10.0
                    y = j/10.0
                    z = k/10.0
                    if (-3.0 < x < -1.0) and (-3.0 < y < -1.0) and (-3.0 < z < -1.0):
                        self.assertTrue(rprism1.contains((x,y,z)))
                    else:
                        self.assertFalse(rprism1.contains((x,y,z)))




if __name__ == '__main__':

    unittest.main()