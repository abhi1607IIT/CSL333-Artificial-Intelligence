import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math
import scipy.io
from sklearn import svm
from sklearn.svm import SVR


GivenSubjectiveScore = scipy.io.loadmat('../Provided_Code/dmos.mat')['dmos']

RefImagName = scipy.io.loadmat('../Live_Database/databaserelease2/refnames_all.mat')['refnames_all']
GivenImg = ["jp2k","jpeg","wn","gblur","fastfading"]
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
count=0
ans =0.0
training = []
trainingsubjectivescore = []
testing = []
testingsubjectivescore = []
for start in range (0,6):
	if start == 0:
		for i in range (1,20):
			img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0)
			img = np.array(img)/255
			local = np.array_str(RefImagName[0][i-1])
			localagain = local[2:len(local)-2]
			imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0)
			imgRef = np.array(imgRef)/255
			u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
			up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
			z= min(img.shape[0],img.shape[1])
			z=300
			feature = np.zeros((1,z), dtype=float)
			for p in range (0,z):
				tempu =0.0
				tempv =0.0
				for q in range (0,z):
					tempu = tempu + u[p][q]*up[p][q]
					tempv = tempv + v[p][q]*vp[p][q]
				feature[0][p]= tempu + tempv
			# print(feature.shape)
			cropimg = np.zeros((z,z),dtype=float)
			for p in range (0,z):
				for q in range (0,z):
					cropimg[p][q]=img[p][q]
			# print(GivenSubjectiveScore[0][i-1])
			if i<15:
				training.append(feature[0])
				trainingsubjectivescore.append(GivenSubjectiveScore[0][i-1])
			else:
				testing.append(feature[0])
				testingsubjectivescore.append(GivenSubjectiveScore[0][i-1])

svr_rbf.fit(training,trainingsubjectivescore)
predictScore = svr_rbf.predict(testing)
print(predictScore)
print(testingsubjectivescore)
	# elif start == 1:
	# 	for i in range (1,233):
	# 		img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0)
	# 		img = np.array(img)/255
	# 		local = np.array_str(RefImagName[0][i-1 + 227])
	# 		localagain = local[2:len(local)-2]
	# 		imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0)
	# 		imgRef = np.array(imgRef)/255
	# 		u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
	# 		up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
	# 		z= min(img.shape[0],img.shape[1])
	# 		feature = np.zeros((z,1), dtype=float)
	# 		for p in range (0,z):
	# 			tempu =0.0
	# 			tempv =0.0
	# 			for q in range (0,z):
	# 				tempu = tempu + u[p][q]*up[p][q]
	# 				tempv = tempv + v[p][q]*vp[p][q]
	# 			feature[p]= tempu + tempv
	# 		# print(feature.shape)
	# else:
	# 	if start == 4:
	# 		for i in range (2,174):
	# 			img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0)
	# 			img = np.array(img)/255
	# 			local = np.array_str(RefImagName[0][i-1 + 982 - 174])
	# 			localagain = local[2:len(local)-2]
	# 			imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0)
	# 			imgRef = np.array(imgRef)/255
	# 			u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
	# 			up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
	# 			z= min(img.shape[0],img.shape[1])
	# 			feature = np.zeros((z,1), dtype=float)
	# 			for p in range (0,z):
	# 				tempu =0.0
	# 				tempv =0.0
	# 				for q in range (0,z):
	# 					tempu = tempu + u[p][q]*up[p][q]
	# 					tempv = tempv + v[p][q]*vp[p][q]
	# 				feature[p]= tempu + tempv
	# 			# print(feature.shape)
	# 	else:
	# 		for i in range (1,174):
	# 			img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0)
	# 			img = np.array(img)/255
	# 			local = np.array_str(RefImagName[0][i-1 + 227 + 233 + count*174])
	# 			localagain = local[2:len(local)-2]
	# 			imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0)
	# 			imgRef = np.array(imgRef)/255
	# 			u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
	# 			up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
	# 			z= min(img.shape[0],img.shape[1])
	# 			feature = np.zeros((z,1), dtype=float)
	# 			for p in range (0,z):
	# 				tempu =0.0
	# 				tempv =0.0
	# 				for q in range (0,z):
	# 					tempu = tempu + u[p][q]*up[p][q]
	# 					tempv = tempv + v[p][q]*vp[p][q]
	# 				feature[p]= tempu + tempv
	# 			count = 1
	# 			# print(feature.shape)
