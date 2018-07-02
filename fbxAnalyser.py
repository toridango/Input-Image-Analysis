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
from pyassimp import helper


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

def getBoundingBox(filepath):
	return pyassimp.helper.get_bounding_box(getScene(filepath))

def challenge(challenger, champion, comp):
	assert comp != None

	if comp == "<" or comp == "lesser":
		return challenger if challenger < champion else champion
	elif comp == ">" or comp == "greater":
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

	#
	# print maxX, minX
	# print maxY, minY
	# print maxZ, minZ

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


def getMinMaxCoords(scene, verbose = False):

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



	if verbose:

		print("      Min    |     Max  ")
		print("X:  {0: <8.2f} | {1: >8.2f}".format(minX, maxX))
		print("Y:  {0: <8.2f} | {1: >8.2f}".format(minY, maxY))
		print("Z:  {0: <8.2f} | {1: >8.2f}".format(minZ, maxZ))


		print ("\nSizes: {0:.2f} x {1:.2f} x {2:.2f}\n".format(width, height, depth))


	# don't forget this one, or you will leak!
	pyassimp.release(scene)

	# /100 to convert from cm to m
	return [minX/100.0, minY/100.0, minZ/100.0], [maxX/100.0, maxY/100.0, maxZ/100.0]


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


def rec_printTree(node, depth = 0):
	for c in node.children:
		print str(c)+"    "*depth
		rec_printTree(c)



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


def explore(fbx_carPath):
	scene = pyassimp.load(fbx_carPath)
	MODEL = fbx_carPath

	#the model we load
	print "MODEL:", MODEL
	print

	#write some statistics
	print "SCENE:"
	print "  meshes:", len(scene.meshes)
	print "  materials:", len(scene.materials)
	print "  textures:", len(scene.textures)
	print

	print "MESHES:"
	for index, mesh in enumerate(scene.meshes):
		print "  MESH", index+1
		print "    material:", mesh.mMaterialIndex+1
		print "    vertices:", len(mesh.vertices)
		print "    first:", mesh.vertices[:3]
		print "    colors:", len(mesh.colors)
		tc = mesh.texcoords
		print "    texture-coords 1:", len(tc[0]), "first:", tc[0][:3]
		print "    texture-coords 2:", len(tc[1]), "first:", tc[1][:3]
		print "    texture-coords 3:", len(tc[2]), "first:", tc[2][:3]
		print "    texture-coords 4:", len(tc[3]), "first:", tc[3][:3]
		print "    uv-component-count:", len(mesh.mNumUVComponents)
		print "    faces:", len(mesh.faces), "first:", [f.indices for f in mesh.faces[:3]]
		print "    bones:", len(mesh.bones), "first:", [b.mName for b in mesh.bones[:3]]
		print

		print "MATERIALS:"
		for index, material in enumerate(scene.materials):
			print "  MATERIAL", index+1
			properties = pyassimp.GetMaterialProperties(material)
			for key in properties:
				print "    %s: %s" % (key, properties[key])
				print

				print "TEXTURES:"
				for index, texture in enumerate(scene.textures):
					print "  TEXTURE", index+1
					print "    width:", texture.mWidth
					print "    height:", texture.mHeight
					print "    hint:", texture.achFormatHint
					print "    data (size):", texture.mWidth*texture.mHeight

					# Finally release the model
					pyassimp.release(scene)



def main(verbose = False):
	fbx_carPath = "./resources/SpeedLimit.fbx"

	# useFBXLibrary(fbx_carPath):
	# readFBX(fbx_carPath)
	scene = getScene(fbx_carPath)
	# getSizes(scene, verbose = verbose)
	# print pyassimp.helper.get_bounding_box(scene)
	print getMinMaxCoords(scene)




if __name__ == '__main__':
	verbose = False

	start_time = time.time()
	main(verbose)
	print("Elapsed time: {} seconds".format(time.time() - start_time))
