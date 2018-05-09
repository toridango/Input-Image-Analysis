import unittest
import pprint

from getParamsFromR import *
from virtObject import *


quickTestsOnly = True

def checkEqualList(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

def checkEqualFloatList(L1, L2, decimals = 5):
    if len(L1) == len(L2):
        diff = np.abs(np.around(L1, decimals) - np.round(L2, decimals))
        # print diff
        return np.count_nonzero(diff) == 0
    else:
        return False


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

    def test_rotate_point(self):


        point = [0, 0, 0]
        pivot = [0, 0, 0]
        correct = [0, 0, 0]

        rot = rotatePoint(point, pivot, yaw = 0, pitch = 0, roll = 0)
        rot = list(rot)
        self.assertTrue(checkEqualFloatList(rot, correct))

        point = [0, 2, 0]
        pivot = [0, 0, 0]
        correct = [2/np.sqrt(2), 2/np.sqrt(2), 0]

        rot = rotatePoint(point, pivot, yaw = -45, pitch = 0, roll = 0)
        rot = list(rot)
        self.assertTrue(checkEqualFloatList(rot, correct))

        point = [0, 3, 0]
        pivot = [0, 1, 0]
        correct = [2/np.sqrt(2), 2/np.sqrt(2) + 1, 0]

        rot = rotatePoint(point, pivot, yaw = -45, pitch = 0, roll = 0)
        rot = list(rot)
        self.assertTrue(checkEqualFloatList(rot, correct))


    def test_rotate_box(self):


        point = [[0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 0, 1]]

        correct = [[0, 0, 0],
                [1, 0, 0],
                [1, -1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, -1, 1],
                [0, -1, 1]]


        rot = rotateBox(point, np.array([0,0,0]), yaw = -90, pitch = 0, roll = 0)
        self.assertTrue(checkEqualFloatList(rot, correct))

        point = [[-1, -1, 0],
                [-1, 1, 0],
                [1, 1, 0],
                [1, -1, 0],
                [-1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
                [1, -1, 1],]

        ts2 = 2/np.sqrt(2)

        correct = [[-ts2, 0, 0],
                [0, ts2, 0],
                [ts2, 0, 0],
                [0, -ts2, 0],
                [-ts2, 0, 1],
                [0, ts2, 1],
                [ts2, 0, 1],
                [0, -ts2, 1]]

        rot = rotateBox(point, np.array([0,0,0]), yaw = -45, pitch = 0, roll = 0)
        self.assertTrue(checkEqualFloatList(rot, correct))



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


    def text_pointInPrism_vs_prefilter(self):


        eightpoints = [[ -3, -1,  0], [ -1, -2,  0], [  0,  0,  0], [ -2,  1,  0],
                        [ -3, -1,  2.24], [ -1, -2,  2.24], [  0,  0,  2.24], [ -2,  1,  2.24]]

        x, y, z = -1.5, -0.5, 0
        width, height, depth = np.sqrt(5), np.sqrt(5), np.sqrt(5) # 2.24, 2.24, 2.24
        box = self.getAbsoluteBoundingBox((x,y,z), (width, height, depth), yaw = 0, pitch = 45, roll = 0)
        rprism = RectPrism(box)

        i, j, k = -0.1, 1, 1

        self.assertTrue(np.logical_and(\
						np.logical_and(\
						np.logical_and(x - width/2.0 < i, i < x + width/2.0),
						np.logical_and(y < j, j < y + height)),
						np.logical_and(z - depth/2.0 < k, k < z + depth/2.0)))

        self.assertFalse(rprism.contains((i,j,k)))





    @unittest.skipIf(quickTestsOnly == True, "Exhaustive tests are deactivated")
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

    @unittest.skipIf(quickTestsOnly == True, "Exhaustive tests are deactivated")
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

    @unittest.skipIf(quickTestsOnly == True, "Exhaustive tests are deactivated")
    def test_extensive_origin_pointInPrism(self):

        eightpoints = [[1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1],
                    [1,1,-1], [1,-1,-1], [-1,-1,-1], [-1,1,-1]]

        rprism1 = RectPrism(eightpoints)

        for i in range(-20, 20):
            for j in range(-20, 20):
                for k in range(-20, 20):
                    x = i/10.0
                    y = j/10.0
                    z = k/10.0
                    if (-1.0 < x < 1.0) and (-1.0 < y < 1.0) and (-1.0 < z < 1.0):
                        self.assertTrue(rprism1.contains((x,y,z)))
                    else:
                        self.assertFalse(rprism1.contains((x,y,z)))



if __name__ == '__main__':

    unittest.main()
