#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import fbx
import codecs
import numpy as np
import time
import sys

'''
https://github.com/assimp/assimp

HOW TO INSTALL (go to master folder):
cmake CMakeLists.txt
Open your default IDE and build it

OR

- Download zip
- Unzip, & cmake CMakeLists.txt
- Open the "native tools VS 2017" command prompt,
- Go to the master folder
- Run: mkdir _build & cd _build & cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=assimp-install -DASSIMP_BUILD_TESTS=OFF -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_TESTS=OFF .. & nmake & nmake install


OR (not tested, maybe cmake missing)

- Run: git clone --depth=1 https://github.com/assimp/assimp.git & cd assimp & mkdir _build & cd _build & cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=assimp-install -DASSIMP_BUILD_TESTS=OFF -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_TESTS=OFF .. & nmake & nmake install


then >port>Pyassimp: python setup.py install
(Requires CMake 2.6+ & Python 2.6+)

NOW Paste DLL (maybe .lib too) in Python27\Lib\site-packages\pyassimp
'''
import pyassimp


def readFBX(filepath):


	with open(filepath, "rb") as binary_file:
	    # Read the whole file at once
	    data = binary_file.read()#.decode("utf-8", "replace")
	    print(codecs.encode(data, 'base64'))
	print(data)


	# with open(filepath, "rb") as f:
	# 	f.seek(0)
	# 	data = byte = f.read(1)
	# 	while byte != "":

	# 		byte = f.read(1)
			# print byte,
			# data += byte



def getScene(filepath):
	return pyassimp.load(filepath)

def challenge(challenger, champion, comp):
	assert comp != None

	if comp == "<" or comp == "lesser":
		return challenger if challenger < champion else champion
	elif comp == ">" or comp == greater:
		return challenger if challenger > champion else champion


def getSizes(scene, verbose = False):

	if verbose:
		print("Meshes")
		print(len(scene.meshes))
		print("Vertices")
		print(len(scene.meshes[0].vertices))

	minX = np.inf
	maxX = -np.inf

	minY = np.inf
	maxY = -np.inf

	minZ = np.inf
	maxZ = -np.inf

	for mesh in scene.meshes:
		for vertex in mesh.vertices:
			x = vertex[0]
			y = vertex[1]
			z = vertex[2]

			minX = challenge(x, minX, "<")
			maxX = challenge(x, maxX, ">")
			minY = challenge(y, minY, "<")
			maxY = challenge(y, maxY, ">")
			minZ = challenge(z, minZ, "<")
			maxZ = challenge(z, maxZ, ">")

		# if x < minX:
		# 	minX = x
		# if x > maxX:
		# 	maxX = x

		# if y < minY:
		# 	minY = y
		# if y > maxY:
		# 	maxY = y

		# if z < minZ:
		# 	minZ = z
		# if z > maxZ:
		# 	maxZ = z

	# /100 to convert from cm to m
	width = (maxX - minX)/100.0
	height = (maxY - minY)/100.0
	depth = (maxZ - minZ)/100.0

	if verbose:

		print("      Min    |     Max  ")
		print("X:  {0: <8.2f} | {1: >8.2f}".format(minX, maxX))
		print("Y:  {0: <8.2f} | {1: >8.2f}".format(minY, maxY))
		print("Z:  {0: <8.2f} | {1: >8.2f}".format(minZ, maxZ))


		print ("\nSizes: {0:.2f} x {1:.2f} x {2:.2f}\n".format(width, height, depth))




	# don't forget this one, or you will leak!
	pyassimp.release(scene)

	return width, height, depth


def exploreNodes(node, depth = 0):

	for i in xrange(node.GetChildCount()):
		nextNode = node.GetChild(i)
		nextCount = nextNode.GetChildCount()

		if nextCount > 0:
			print("  "*depth, nextNode.GetName(), ":", nextNode.GetChildCount())
		else:
			# print "  "*depth, nextNode.EvaluateGlobalTransform().GetT()
			print("  "*depth,nextNode.GetGeometry().BBoxMin)

		exploreNodes(nextNode, depth = depth+1)


def useFBXLibrary(fbx_carPath):

	manager = fbx.FbxManager.Create()

	importer = fbx.FbxImporter.Create(manager, "FbxImporter")

	status = importer.Initialize(fbx_carPath)

	assert status != False, "FbxImporter initialisation failed\nError: {}".format(importer.GetLastErrorString())

	scene = fbx.FbxScene.Create(manager, "ObjScene")

	importer.Import(scene)
	importer.Destroy()


	rootNode = scene.GetRootNode()

	exploreNodes(rootNode, depth = 0)

	# rootNode.GetGeometry().ComputeBBox()


def main(verbose = False):
	fbx_carPath = "./resources/some_car.fbx"

	# useFBXLibrary(fbx_carPath):
	# readFBX(fbx_carPath)
	getSizes(getScene(fbx_carPath), verbose = verbose)





if __name__ == '__main__':
	verbose = False

	start_time = time.time()
	main(verbose)
	print("Elapsed time: {} seconds".format(time.time() - start_time))
