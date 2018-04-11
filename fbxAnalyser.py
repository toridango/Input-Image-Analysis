#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fbx



def readFBX(filepath):


	with open(filepath, "rb") as binary_file:
	    # Read the whole file at once
	    data = binary_file.read()
	    print(data.decode('utf-8'))


	# with open(filepath, "rb") as f:
	# 	f.seek(0)
	# 	data = byte = f.read(1)
	# 	while byte != "":
	# 		byte = f.read(1)
	# 		print byte,
	# 		data += byte



def exploreNodes(node, depth = 0):	

	for i in xrange(node.GetChildCount()):
		nextNode = node.GetChild(i)
		nextCount = nextNode.GetChildCount()

		if nextCount > 0:
			print "  "*depth, nextNode.GetName(), ":", nextNode.GetChildCount()
		else:
			# print "  "*depth, nextNode.EvaluateGlobalTransform().GetT()
			print "  "*depth,nextNode.GetGeometry().BBoxMin

		exploreNodes(nextNode, depth = depth+1)


def useFBXLibrary():

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


def main():
	fbx_carPath = "./resources/some_car.fbx"
	
	readFBX(fbx_carPath)
	




if __name__ == '__main__':
	main()