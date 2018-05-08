import numpy as np


class RectPrism(object):

    '''
    octapoint:
    list with 8 sub-lists of x, y, z
    '''
    def __init__(self, eightpoints):

        '''
        Indices of the vertices
           5-------6
          /|      /|
         / |     / |
        4--|----7  |
        |  1----|--2
        | /     | /
        0-------3
        '''
        self.vertices = np.array(eightpoints)

        # Edges of the rectangular prism
        self.u = self.vertices[1] - self.vertices[0]
        self.v = self.vertices[3] - self.vertices[0]
        self.w = self.vertices[4] - self.vertices[0]

    '''
    Checks if point is inside the rectangular prism. If it is exactly
    on a face of the prism or outside the volume, it returns False. Otherwise,
    it returns True.

    point is 3D point to check: (x, y, z)

    Note: tests suggest that this function works even if the vertices were
    input in the wrong order. This can't be possible
    '''
    def contains(self, point):

        if type(point) != np.ndarray:
            point = np.array(point)

        if np.shape(point) != (3):
            point = point.reshape((3))

        upoint = np.dot(self.u, point)
        vpoint = np.dot(self.v, point)
        wpoint = np.dot(self.w, point)

        up0 = np.dot(self.u, self.vertices[0])
        up1 = np.dot(self.u, self.vertices[1])
        vp0 = np.dot(self.v, self.vertices[0])
        vp3 = np.dot(self.v, self.vertices[3])
        wp0 = np.dot(self.w, self.vertices[0])
        wp4 = np.dot(self.w, self.vertices[4])

        # print upoint, vpoint, wpoint, "//",(up0 > upoint > up1), (vp0 > vpoint > vp3), (wp0 > wpoint > wp4), "//", "(", up0, up1, ")", "(", vp0, vp3, ")", "(", wp0, wp4, ")"

        return (up0 > upoint > up1) and (vp0 > vpoint > vp3) and (wp0 > wpoint > wp4)








if __name__ == '__main__':
    main()
