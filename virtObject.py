import numpy as np



def getRotationMatrix(yaw = 0, pitch = 0, roll = 0, rads = False):

	R = np.identity(3)

	if not rads:
		roll = np.deg2rad(roll)
		pitch = np.deg2rad(pitch)
		yaw = np.deg2rad(yaw)

	cosV = np.cos(pitch)
	cosW = np.cos(yaw)
	cosU = np.cos(roll)
	sinV = np.sin(pitch)
	sinW = np.sin(yaw)
	sinU = np.sin(roll)

	R[0,0] = cosV*cosW
	R[0,1] = sinU*sinV*cosW - cosU*sinW
	R[0,2] = sinU*sinW + cosU*sinV*cosW

	R[1,0] = cosV*sinW
	R[1,1] = cosU*cosW + sinU*sinV*sinW
	R[1,2] = cosU*sinV*sinW - sinU*cosW

	R[2,0] = -sinV
	R[2,1] = sinU*cosV
	R[2,2] = cosU*cosV

	return R


'''
	point: [x, y, z] that rotats
	pivot: [x, y, z] around which "point" rotates

	( angles in degrees. If they're in radians, set rads to True in function call )
	yaw: rotation around z axis
	pitch: rotation around y axis
	roll: rotation around x axis

	returns: rotated point
'''
def rotatePoint(point, pivot, yaw = 0, pitch = 0, roll = 0, rads = False):
	point = np.transpose(np.matrix(point))
	pivot = np.transpose(np.matrix(pivot))


	p_prime = point - pivot
	p_prime = getRotationMatrix(yaw, pitch, roll, rads = rads) * p_prime
	p_prime += pivot


	return np.array(p_prime).flatten()


def rotateBox(box, pivot, yaw = 0, pitch = 0, roll = 0):

	rot = []

	for i, point in enumerate(box):
		rot.append(list(rotatePoint(point, pivot, yaw, pitch, roll)))

	return rot



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
        self.i = self.vertices[1] - self.vertices[0]
        self.j = self.vertices[3] - self.vertices[0]
        self.k = self.vertices[4] - self.vertices[0]

        # Edges of the rectangular prism
        self.u = self.vertices[0] - self.vertices[1]
        self.v = self.vertices[0] - self.vertices[3]
        self.w = self.vertices[0] - self.vertices[4]

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

        return (up0 > upoint) and (upoint > up1) and (vp0 > vpoint) and (vpoint > vp3) and (wp0 > wpoint) and (wpoint > wp4)
        # return (up0 < upoint) and (upoint < up1) and (vp0 < vpoint) and (vpoint < vp3) and (wp0 < wpoint) and (wpoint < wp4)

    '''
    Checks if point is inside the rectangular prism. If it is exactly
    on a face of the prism or outside the volume, it returns False. Otherwise,
    it returns True.

    point is 3D point to check: (x, y, z)

    Note: tests suggest that this function works even if the vertices were
    input in the wrong order. This can't be possible
    '''
    def contains_new(self, point):

        if type(point) != np.ndarray:
            point = np.array(point)

        if np.shape(point) != (3):
            point = point.reshape((3))

        v = point - self.vertices[0]

        ii = np.dot(self.i, self.i)
        jj = np.dot(self.j, self.j)
        kk = np.dot(self.k, self.k)
        vi = np.dot(v, self.i)
        vj = np.dot(v, self.j)
        vk = np.dot(v, self.k)

        # print upoint, vpoint, wpoint, "//",(up0 > upoint > up1), (vp0 > vpoint > vp3), (wp0 > wpoint > wp4), "//", "(", up0, up1, ")", "(", vp0, vp3, ")", "(", wp0, wp4, ")"

        return (0 < vi < ii) and (0 < vj < jj) and (0 < vk < kk)






if __name__ == '__main__':
    main()
