#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..\\cityscapesScripts-master\\cityscapesscripts\\helpers')

import cv2
import numpy as np
from numpy.matlib import repmat
import imutils
import json
from labels import *
from pprint import pprint



CTS_helpers_relPath = "\\cityscapesScripts-master\\cityscapesscripts\\helpers\\"


def getLabeledPixels(semantic, target, label, show = True):

	# create image with only pixels of one label
	mask = cv2.inRange(semantic, name2label[label].color, name2label[label].color)
	output = cv2.bitwise_and(target, target, mask = mask)

	if show:
		cv2.imshow('masked', output)
		cv2.waitKey(0)

	return output




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

		self.semantic = None
		self.depthImg = None

		self.imgRecord = {}
		self.sem_info = {}
		self.centroid = {}
		self.camFlag = False



	def loadImages(self):
		self.rgb = cv2.imread("\\".join([self.path, self.imagePath, self.split, self.city, self.imgName+self.rgb_suffix]))
		self.imgRecord["rgb"] = 1
		self.semantic = cv2.imread("\\".join([self.path, self.gtFinePath, self.split, self.city, self.imgName+self.sem_suffix]))
		self.imgRecord['semantic'] = 1
		self.disparity = cv2.imread("\\".join([self.path, self.disparityPath, self.split, self.city, self.imgName+self.dis_suffix]), , cv2.IMREAD_GRAYSCALE)
		self.imgRecord['disparity'] = 1
		# cv2.imshow('img', self.disparity)
		# cv2.waitKey(0)



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

		self.centroid['road'] = self.getCentroidOfLabel('road')

		with open("\\".join([self.path, self.cameraPath, self.split, self.city, self.imgName+self.cam_suffix])) as f:
			self.camera = json.load(f)
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

		baseline = self.camera["extrinsic"]["baseline"]
		focal = self.camera["intrinsic"]["fx"]

		# change to float to be able to save decoded values
		self.disparity = self.disparity.astype(float)

		for i, row in enumerate(self.disparity):
			for j, elem in enumerate(row):
				if elem > 0:
					self.disparity[i,j] = (float(elem) - 1.0) / 256.0

		self.depthImg = (baseline * focal) / self.disparity

		# takes care of infinites, and allows normalisation
		for i, row in enumerate(self.depthImg):
			for j, elem in enumerate(row):
				if elem == np.inf:
					self.depthImg[i,j] = 0.0

		# normalise
		self.depthImg = self.depthImg/self.depthImg.max()

		if verbose:
			cv2.imshow("img", self.depthImg)
			cv2.waitKey(0)

		self.imgRecord["depth"] = 1

		return self.depthImg*255.0


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



def save_ply(ply_file, points):

	with open(ply_file,"w") as file:
			file.write('''ply
		format ascii 1.0
		element vertex %d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		property uchar alpha
		end_header
		%s
		'''%(len(points),"".join(points)))
			file.close()


	
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
	
	imgset = ImgSet(split, city, imgName, pathDict)
	imgset.loadImages()

	imgset.getRoadInfo(imgset.semantic)
	# kernel = np.ones((50,50), np.uint8)

	# kernel = np.array( [[0,0,0,0,0],
	# 					[0,0,1,0,0],
	# 					[0,0,1,0,0],
	# 					[0,0,1,0,0],
	# 					[1,1,1,1,1]])

	# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (50, 50))

	# roadLabel = getLabeledPixels(imgset.semantic, imgset.semantic, 'road', False)

	# placeable = cv2.morphologyEx(roadLabel, cv2.MORPH_OPEN, kernel)

	# placeable = cv2.erode(getLabeledPixels(placeable, placeable, 'road', False),kernel,iterations = 3)

	# cv2.imshow('img', imgset.semantic + placeable)
	# cv2.waitKey(0)


if __name__ == '__main__':
	main()


	# Z (depth) = (focalLength * baseline) / disparity