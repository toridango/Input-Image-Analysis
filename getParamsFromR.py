#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..\\cityscapesScripts-master\\cityscapesscripts\\helpers')
import time

import cv2
import numpy as np
from numpy.matlib import repmat
import imutils
import json
from labels import *
from pprint import pprint
import random as rand

from fbxAnalyser import *
from virtObject import *

# k-d tree example https://github.com/tsoding/kdtree-in-python/blob/master/main.py

CTS_helpers_relPath = "\\cityscapesScripts-master\\cityscapesscripts\\helpers\\"

'''
semantic: semantic segmentation image
target: image on which to apply the mask
label: string containing the name of the cityscapes label to keep
'''
def getLabeledPixels(semantic, target, label, show = True):

	# find the colors within the specified boundaries and apply the mask
	# ex: mask = cv2.inRange(image, lower, upper)
	mask = cv2.inRange(semantic, name2label[label].color, name2label[label].color)
	output = cv2.bitwise_and(target, target, mask = mask)

	if show:
		cv2.imshow('masked', output)
		cv2.waitKey(0)

	return output



def construct_ply_header(len_points, alpha = False):
	"""Generates a PLY header given a total number of 3D points and
	coloring property if specified
	"""
	header = ['ply',
			  'format ascii 1.0',
			  'element vertex {}',
			  'property float x',
			  'property float y',
			  'property float z',
			  'property uchar red',
			  'property uchar green',
			  'property uchar blue',
			  'property uchar alpha',
			  'end_header']
	if not alpha:
		return '\n'.join(header[0:9] + [header[-1]]).format(len_points)

	return '\n'.join(header).format(len_points)


def points_to_string(points):
	return '\n'.join(['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f}'.format(*p) for p in points.tolist()])


def save_ply(ply_file, points):

	with open(ply_file, 'w') as f:
		f.write('\n'.join([construct_ply_header(points.shape[0]), points_to_string(points)]))



def getIntrinsicMatrix(cameraDict):

	fx = cameraDict["intrinsic"]["fx"]
	fy = cameraDict["intrinsic"]["fy"]
	cx = cameraDict["intrinsic"]["u0"]
	cy = cameraDict["intrinsic"]["v0"]

	return np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def getExtrinsicMatrix(cameraDict):


	x = cameraDict["extrinsic"]["x"]
	y = cameraDict["extrinsic"]["y"]
	z = cameraDict["extrinsic"]["z"]

	pitch = cameraDict["extrinsic"]["pitch"]
	yaw = cameraDict["extrinsic"]["yaw"]
	roll = cameraDict["extrinsic"]["roll"]

	cosP = np.cos(pitch)
	cosY = np.cos(yaw)
	cosR = np.cos(roll)
	sinP = np.sin(pitch)
	sinY = np.sin(yaw)
	sinR = np.sin(roll)

	Rt = np.identity(4)

	Rt[0,0] = cosY*cosP
	Rt[0,1] = (cosY*sinP*sinR) - (sinY*cosR)
	Rt[0,2] = (cosY*sinP*cosR) + (sinY*sinR)
	Rt[0,3] = x

	Rt[1,0] = sinY*cosP
	Rt[1,1] = (sinY*sinP*sinR) + (cosY*cosR)
	Rt[1,2] = (sinY*sinP*cosR) - (cosY*sinR)
	Rt[1,3] = y

	Rt[2,0] = -sinP
	Rt[2,1] = cosP*sinR
	Rt[2,2] = cosP*cosR
	Rt[2,3] = z

	return Rt


'''
RGB tuple = (r, g, b)

return list of labels with that colour
'''
def colour2labels(RGB):

	if RGB == (0,0,0):
		return None

	return list(filter(lambda name: name2label[name].color == RGB, name2label))



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


class ImgSet(object):


	def __init__(self, split, city, imgName, pathDict):

		self.path = ".."

		self.disparityPath = pathDict["disparityPath"] #'Cityscapes_Disparity\\disparity_trainvaltest\\disparity'
		self.cameraPath = pathDict["cameraPath"] #'Cityscapes_Disparity\\camera_trainvaltest\\camera'
		self.gtFinePath = pathDict["gtFinePath"] #'gtFine_trainvaltest\\gtFine'
		self.imagePath = pathDict["imagePath"] #'leftImg8bit_trainvaltest\\leftImg8bit'

		self.split = split #'train'
		self.city = city #'jena'
		self.imgName = imgName #'jena_000000_000019'

		self.rgb_suffix = "_leftImg8bit.png"
		self.sem_suffix = "_gtFine_color.png"
		self.dis_suffix = "_disparity.png"
		self.cam_suffix = "_camera.json"

		self.rgb = None
		self.semantic = None
		self.depthImg = None
		self.K = None
		self.Rt = None

		self.imgRecord = {}
		self.sem_info = {}
		self.centroid = {}
		self.camFlag = False
		self.pointCloudSaved = False

		self.pointCloud = None



	def loadImages(self):

		sys.stdout.write('Reading Images for {}...'.format(self.imgName))

		self.rgb = cv2.imread("\\".join([self.path, self.imagePath, self.split, self.city, self.imgName+self.rgb_suffix]))
		self.imgRecord['rgb'] = 1

		self.semantic = cv2.imread("\\".join([self.path, self.gtFinePath, self.split, self.city, self.imgName+self.sem_suffix]))
		self.imgRecord['semantic'] = 1

		# It seems like the flags -1 and cv2.IMREAD_ANYDEPTH have the same effect in the end
		# for the images being used. ANYDEPTH has been left since it should be more flexible
		self.disparity = cv2.imread("\\".join([self.path, self.disparityPath, self.split, self.city, self.imgName+self.dis_suffix]), cv2.IMREAD_ANYDEPTH)
		self.imgRecord['disparity'] = 1

		# cv2.imshow('img', self.disparity)
		# cv2.waitKey(0)

		sys.stdout.write(' Done\n')


	'''
	param:
		label - string containing label of desired points
	returns:
		points (in matrix form), pertaining to the targeted label
	'''
	def getPointsWhereLabel(self, label):
		assert(self.pointCloudSaved), "Calling getPointsWhereLabel without having computed the point cloud"
		assert(label in name2label), "Label doesn't exist. Consult cityscapes label information"

		colour = name2label[label].color
		pc = np.asarray(self.pointCloud)

		return pc[((pc[:,3] == colour[0])&(pc[:,4] == colour[1])&(pc[:,5] == colour[2]))]




	'''
	Legacy function to find the centroid of the polygon
	that corresponds to a label in the semantic segmentation
	image
	'''
	def getCentroidOfLabel(self, label, debug = False):

		assert label in name2label, "%r is not a cityscapes label" %label

		labelPix = getLabeledPixels(self.semantic, self.semantic, label, False)
		gray = cv2.cvtColor(labelPix, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]


		M = cv2.moments(cnts[0])
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		if debug:
			img = self.semantic.copy()

			# draw the contour and center of the shape on the image
			cv2.drawContours(img, [cnts[0]], -1, (0, 255, 0), 2)
			cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
			cv2.putText(img, "center", (cX - 20, cY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

			# show the image
			cv2.imshow("Image", img)
			cv2.waitKey(0)

		if len(cnts) == 1:
			return (cX, cY)
		else:
			return None



	def getRoadInfo(self, img):

		assert self.imgRecord['semantic'] == 1, 'Semantic Image not loaded'

		# self.centroid['road'] = self.getCentroidOfLabel('road')

		with open("\\".join([self.path, self.cameraPath, self.split, self.city, self.imgName+self.cam_suffix])) as f:
			self.camera = json.load(f)
			self.K = getIntrinsicMatrix(self.camera)
			self.Rt = getExtrinsicMatrix(self.camera)
			self.camFlag = True
			# pprint(self.camera)

		# cv2.rectangle(self.semantic, (center[0]-100, center[1]-100), (center[0]+100, center[1]+100), (255,0,0))



	def depthFromDisparity(self, verbose = False):
		'''
		disparity img info: precomputed disparity depth maps. To obtain the disparity values,
		compute for each pixel p with p > 0: d = ( float(p) - 1. ) / 256., while a
		value p = 0 is an invalid measurement. Warning: the images are stored as
		16-bit pngs, which is non-standard and not supported by all libraries.

		baseline in meters
		focal in pixels
		returns depth image in range 0-255
		'''

		assert self.camFlag == True, "Camera information not loaded"
		assert self.imgRecord["disparity"] == 1, "Disparity image not loaded"

		sys.stdout.write('Computing depth...')

		baseline = self.camera["extrinsic"]["baseline"]
		focal = self.camera["intrinsic"]["fx"]

		# change to float to be able to save decoded values
		self.disparity = self.disparity.astype(float)

		# commenting this (below) double loop changes value
		# of width of closest car (jena 0) from 450 to 1.5 (aprox)

		for i, row in enumerate(self.disparity):
			for j, elem in enumerate(row):
				if elem > 0:
					self.disparity[i,j] = (elem - 1.0) / 256.0

		self.depthImg = (baseline * focal) / (self.disparity + 0.0000000001) # small value to avoid DIV 0

		# takes care of infinites, and allows normalisation
		# for i, row in enumerate(self.depthImg):
		# 	for j, elem in enumerate(row):
		# 		if elem == np.inf:
		# 			self.depthImg[i,j] = 0.0

		# normalise
		# self.depthImg = self.depthImg/self.depthImg.max()

		if verbose:
			cv2.imshow("img", self.depthImg)
			cv2.waitKey(0)

		self.imgRecord["depth"] = 1

		# self.depthImg *= 255.0

		sys.stdout.write(' Done\n')


	# legacy iterative method
	def getPointCloud(self, ply_file):

		assert self.camFlag == True, "Camera information not loaded"
		assert self.imgRecord["rgb"] == 1, "RGB Image information not loaded"
		assert self.imgRecord["depth"] == 1, "Depth Image information not computed"

		fx = self.camera["intrinsic"]["fx"]
		fy = self.camera["intrinsic"]["fy"]
		cx = self.camera["intrinsic"]["u0"]
		cy = self.camera["intrinsic"]["v0"]

		points = []
		height = self.rgb.shape[0] # shape[0] is height, shape[1] is width

		for i, row in enumerate(self.depthImg):
			for j, elem in enumerate(row):
				colour = self.rgb[i,j]
				Z = self.depthImg[i,j]
				X = (Z / fx) * (i - cx)
				Y = (Z / fy) * (j - cy) # need to invert
				Y = height - Y
				points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,colour[2],colour[1],colour[0]))

		save_ply(ply_file, points)


	def getPointCloudMatricial(self, colourSource='semantic', max_depth=20000.0, full_transform = False, verbose = False):

		colourImage = None

		# Warning:
		# colourImage is pointing to the intended image, and so the latter will be deformed
		if colourSource == 'rgb':
			colourImage = self.rgb
		else:
			colourImage = self.semantic


		assert colourImage is not None, "Invalid colour source"
		assert self.imgRecord["depth"] == 1, "Depth image not obtained"
		assert self.camFlag == True, "Camera parameters not loaded"

		sys.stdout.write('Computing Point Cloud...')

		# shape[0] is height, shape[1] is width

		# Get number of pixels
		pixel_length = colourImage.shape[1] * colourImage.shape[0]

		# IMPORTANT: Order of Xs and Ys is inverted, probably to correct
		# the inversion that results from the 2D to 3D transform
		# (based on Carla's depth to local point cloud conversion)

		# Prepare Xs in a 1D array
		u_coords = repmat(np.r_[colourImage.shape[1]-1:-1:-1], colourImage.shape[0], 1).reshape(pixel_length)

		# Prepare Ys in a 1D array
		v_coords = repmat(np.c_[colourImage.shape[0]-1:-1:-1], 1, colourImage.shape[1]).reshape(pixel_length)

		# Reshape colour image into colour-trio array
		colourImage = colourImage.reshape(pixel_length, 3)
		# Reshape depth image
		self.depthImg = np.reshape(self.depthImg, pixel_length)

		# print depthImage.shape, u_coords.shape, v_coords.shape, colourImage.shape

		# Search for pixels where the depth is greater than max_depth to
		# delete them
		max_depth_indexes = np.where(self.depthImg > max_depth)

		self.depthImg = np.delete(self.depthImg, max_depth_indexes)
		u_coords = np.delete(u_coords, max_depth_indexes)
		v_coords = np.delete(v_coords, max_depth_indexes)
		colourImage = np.delete(colourImage, max_depth_indexes, axis=0)


		# pd2 = [u,v,1]
		p2d = np.array([u_coords, v_coords, np.ones_like(u_coords)])

		if verbose:
			# Should have 3xN
			print "\nP2D:", p2d.shape

		# K-1 · list of u,v,1 gives us the 2D to 3D transform (depth still missing)
		# shapes: 3x3 · 3xN
		# P = [X,Y,Z]
		p3d = np.dot(np.linalg.inv(self.K), p2d)

		if verbose:
			# Should still be 3xN
			print "P3D:", p3d.shape
			print "* Depth:", self.depthImg.shape

		# Rt-1 here?
		# raw shapes: 4x4 · 3xN  //  we want to end up with 3xN

		# element-wise multiplication to apply the depth
		p3d = np.multiply(p3d, self.depthImg)

		if verbose:
			print "P3D after depth:", p3d.shape
			# print p3d
			print "concat Colour:", colourImage.shape

		# Transpose into Nx3 and concatenate the colours to get Nx6
		self.pointCloud = np.concatenate((p3d.T, colourImage[:,::-1]), axis = 1)
		self.pointCloudSaved = True


		if verbose:
			print self.pointCloud

		sys.stdout.write(' Done\n')

		return self.pointCloud


	'''
	point: (x, y)
	sizes: (w, h, d)

	exclude: list of labels to exclude from the search
	'''
	def objectOnPointCollides(self, point, sizes, exclude = []):

		x, y = point
		w, h, d = sizes

		pass



	def PC_to_IS(self, x, y, z):
		'''
		Convert from the generated point cloud coordinates
		to International System (used later in Unity)
		'''
		pass


	def IS_to_PC(self, x, y, z):
		'''
		Convert from International System coordinates to
		the ones used in the generated point cloud
		'''
		pass


	'''
	x, y, z: point on which the object will rest
			(centered on its x and z axes):
	w, h, d: dimensions of bounding box

	yaw: angle around y. 0 considered pointing towards positive z (not verified)

	returns: box in the shape of
	           5-------6		+y
	          /|      /|		|  +z
	         / |     / |		| /
	        4-------7  |  +x ----/
	        |  1----|--2
	        | /   c | /
	        0-------3
	where c is the point (x,y,z) on which the box is being placed
	(coordinates in respect to the camera)
	'''
	def getAbsoluteBoundingBox(self, (x,y,z), (w,h,d), yaw = 0, pitch = 0, roll = 0):
		# return [[x - w/2.0, x + w/2.0],
		# 		[    y    ,  y  +  h ],
		# 		[z - d/2.0, z + d/2.0]]

		print x,y,z,"//",w,h,d

												 #    x y z
		box =  [[x + w/2.0,     y, z - d/2.0],	 # 0  + = -
				[x + w/2.0,     y, z + d/2.0],	 # 1  + = +
				[x - w/2.0,     y, z + d/2.0],	 # 2  - = +
				[x - w/2.0,     y, z - d/2.0],	 # 3  - = -
				[x + w/2.0, y + h, z - d/2.0],	 # 4  + + -
				[x + w/2.0, y + h, z + d/2.0],	 # 5  + + +
				[x - w/2.0, y + h, z + d/2.0],	 # 6  - + +
				[x - w/2.0, y + h, z - d/2.0]]	 # 7  - + -

		box = rotateBox(box, [x,y,z], yaw = yaw, pitch = pitch, roll = roll)

		return box

	'''
	Given a list of wanted labels, it finds the points that have them and returns
	their indices in the stored point cloud
	'''
	def findLabelsInPointCloud(self, labelList, verbose = False):

		assert self.pointCloudSaved == True, "Invalid colour source"
		for label in labelList:
			assert label in name2label, "%r is not a cityscapes label" %label

		if verbose:
			print "Finding points with labels:"

		candidateIndexes = np.where(np.all(self.pointCloud[:,3:] == name2label[labelList[0]].color, axis=1))[0]

		if verbose:
			print labelList[0], type(candidateIndexes), candidateIndexes.shape

		for label in labelList[1:]:
			moreCandidates = np.where(np.all(self.pointCloud[:,3:] == name2label[label].color, axis=1))[0]
			candidateIndexes = np.concatenate((candidateIndexes, moreCandidates), axis=0)

			if verbose:
				print label, type(moreCandidates), moreCandidates.shape, "// total", type(candidateIndexes), candidateIndexes.shape

		return candidateIndexes

	'''
		objSizes[0] = width
				[1] = height
				[2] = depth

		yaw: rotation around z axis
		pitch: rotation around y axis
		roll: rotation around x axis

		labelList: list containing labels on which the object can be placed
	'''
	def assignPlacement(self, (width, height, depth), labelList, visualise = True):

		assert self.pointCloudSaved == True, "Point Cloud not found"

		candidateIndexes = self.findLabelsInPointCloud(labelList)
		low = 0
		high = candidateIndexes.shape[0] - 1
		approved = False

		while not approved:

			# choose a random index of the ones with the wanted labels
			choice = int((high-low)*rand.random()+low)
			(x,y,z) = self.pointCloud[candidateIndexes[choice], :3].tolist()[0]
			box = self.getAbsoluteBoundingBox((x,y,z), (width, height, depth), yaw = 0, pitch = 45, roll = 0)
			rprism = RectPrism(box)

			complementaryIndices = np.delete(np.array(xrange(self.pointCloud.shape[0])), candidateIndexes)
			print "Complementary:",complementaryIndices.shape
			# TODO check if points of other labels collide (are inside)
			# If they aren't, approve it and return placement

			# containsAny = np.vectorize(rprism.contains)

			# filter between maxs and mins # TODO TEST THIS vs Contains
			pfIndices = np.logical_and(\
								np.logical_and(\
								np.logical_and(x - width/2.0 < self.pointCloud[complementaryIndices,0], self.pointCloud[complementaryIndices,0] < x + width/2.0),
								np.logical_and(y < self.pointCloud[complementaryIndices,1], self.pointCloud[complementaryIndices,1] < y + height)),
								np.logical_and(z - depth/2.0 < self.pointCloud[complementaryIndices,2], self.pointCloud[complementaryIndices,2] < z + depth/2.0))
			print "Prefilter:", pfIndices.shape
			pfIndices = np.array(pfIndices).flatten()
			print "Flat Prefilter:", pfIndices.shape

			vis = visualise
			if vis and pfIndices.shape[0] != 0:
				aux = box[:]
				for i, e in enumerate(box):
					aux[i] = e + [220, 20, 60]
				save_ply(".\\output\\"+"prefilter_and_wBox.ply", np.concatenate((self.pointCloud[complementaryIndices][pfIndices], np.array(aux)), axis=0))
				vis = False

			print "PC shapes: original:", self.pointCloud.shape, "// complementary", self.pointCloud[complementaryIndices].shape, "// preFilter", self.pointCloud[complementaryIndices][pfIndices].shape


			if self.pointCloud[complementaryIndices][pfIndices].shape[0] == 0:
				approved = True
			else:
				print "Prefilter detected something..."#,self.pointCloud[complementaryIndices,:3][pfIndices]
				inTheBox = np.where(np.any(map(rprism.contains, self.pointCloud[complementaryIndices,:3][pfIndices]), axis=0))[0]
				# print self.pointCloud[complementaryIndices,:3]
				# print self.pointCloud[pfIndices,:3]
				print "Using contains:", inTheBox.shape
				approved = (inTheBox.shape[0] == 0)
				print "Approved:", approved

			# approved = True
			# inTheBox = np.where(np.all(containsAny(self.pointCloud[complementaryIndices,:3]), axis=0))[0]


		if visualise:
			for i, e in enumerate(box):
				aux = box[:]
				for i, e in enumerate(box):
					aux[i] = e + [220, 20, 60]
			save_ply(".\\output\\"+"pointCloud_wBox.ply", np.concatenate((self.pointCloud, np.array(aux)), axis=0))
			save_ply(".\\output\\"+"filter_and_wBox.ply", np.concatenate((self.pointCloud[complementaryIndices][pfIndices], np.array(aux)), axis=0))


	def checkObjectCollision(self, object, complementaryIndices):
		pass




def main():

	split = 'train'
	city = 'jena'
	imgName = 'jena_000000_000019'

	pathDict = {
		"disparityPath":'Cityscapes_Disparity\\disparity_trainvaltest\\disparity',
		"cameraPath": 'Cityscapes_Disparity\\camera_trainvaltest\\camera',
		"gtFinePath": 'gtFine_trainvaltest\\gtFine',
		"imagePath": 'leftImg8bit_trainvaltest\\leftImg8bit'
	}

	objPathDict = {
		"CC3": "./resources/some_car.fbx"
	}

	imgset = ImgSet(split, city, imgName, pathDict)
	imgset.loadImages()

	imgset.getRoadInfo(imgset.semantic)


	imgset.depthFromDisparity()

	'''
	use colourSource argument (string) to choose point coloration in the point cloud
	default: "semantic"
	options: "rgb"
	'''
	points = imgset.getPointCloudMatricial(colourSource = "semantic")

	# save_ply(".\\output\\"+"_".join([split, imgName])+".ply", points)

	w, h, d = getSizes(getScene(objPathDict["CC3"]))

	imgset.assignPlacement((w,h,d),["road"])





if __name__ == '__main__':
	start_time = time.time()
	main()
	print("Elapsed time: {} seconds\n\n".format(time.time() - start_time))


	# Z (depth) = (focalLength * baseline) / disparity

	'''
	Notation for object placement

	Object:
		name
		(allowed) labels
		(allowed) orientations:
			any
			any lane
			right lane
			sign
		constraints:
			any (placement)
			border
			between (close to all the labels)
		special flags:
			driving (has to be on the right lane)
	'''
