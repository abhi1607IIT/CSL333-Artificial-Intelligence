import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math
import scipy.io
from sklearn import svm
from sklearn.svm import SVR
import csv


index = 500

GivenSubjectiveScore = scipy.io.loadmat('../Provided_Code/dmos.mat')['dmos']  # opened dmos.mat file

RefImagName = scipy.io.loadmat('../Live_Database/databaserelease2/refnames_all.mat')['refnames_all'] # opened reference image.mat file
GivenImg = ["jp2k","jpeg","wn","gblur","fastfading"]
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)          # using rbf kernel
count=0
ans =0.0
training = []
trainingsubjectivescore = []
testing = []
testingsubjectivescore = []
imagename = []
destortiontype = []
for start in range (0,5):
	if start == 0:
		for i in range (1,227):
			img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0) # opening 1st folder ie Jp2K
			img = np.array(img)/255
			local = np.array_str(RefImagName[0][i-1])
			localagain = local[2:len(local)-2]
			imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0) #opening refernce images of JP2K folder
			imgRef = np.array(imgRef)/255
			u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)            # SVD 
			up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
			z= min(img.shape[0],img.shape[1])
			index = min(index,img.shape[0],img.shape[1])
			z=438
			feature = np.zeros((1,z), dtype=float)
			for p in range (0,z):
				tempu =0.0
				tempv =0.0
				for q in range (0,z):
					tempu = tempu + u[p][q]*up[p][q]             # calculating feature matrix using x= ai + bi where ai = u . up and bi = v. vp
					tempv = tempv + v[p][q]*vp[p][q]
				feature[0][p]= tempu + tempv
			# print(feature.shape)
			# print(GivenSubjectiveScore[0][i-1])
			if i<180:
				training.append(feature[0])                                    # calculating training featues using K-fold 
				trainingsubjectivescore.append(GivenSubjectiveScore[0][i-1])
			else:
				testing.append(feature[0])
				testingsubjectivescore.append(GivenSubjectiveScore[0][i-1])    # calculating testing featues using K-fold
				destortiontype.append('jp2k')
				imagename.append(localagain)

	elif start == 1:                                              # repeat for 2nd folder and so on
		for i in range (1,233):
			img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0)
			img = np.array(img)/255
			local = np.array_str(RefImagName[0][i-1 + 227])
			localagain = local[2:len(local)-2]
			imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0)
			imgRef = np.array(imgRef)/255
			u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
			up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
			z= min(img.shape[0],img.shape[1])
			index = min(index,img.shape[0],img.shape[1])
			z=438
			feature = np.zeros((1,z), dtype=float)
			for p in range (0,z):
				tempu =0.0
				tempv =0.0
				for q in range (0,z):
					tempu = tempu + u[p][q]*up[p][q]
					tempv = tempv + v[p][q]*vp[p][q]
				feature[0][p]= tempu + tempv
			# print(feature.shape)
			# print(GivenSubjectiveScore[0][i-1])
			if i<186:
				training.append(feature[0])
				trainingsubjectivescore.append(GivenSubjectiveScore[0][i-1 + 227])
			else:
				testing.append(feature[0])
				testingsubjectivescore.append(GivenSubjectiveScore[0][i-1 + 227])
				destortiontype.append('jpeg')
				imagename.append(localagain)
			# print(feature.shape)
	else:
		if start == 4:
			for i in range (2,174):
				# print(i)
				img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0)
				img = np.array(img)/255
				local = np.array_str(RefImagName[0][i-1 + 982 - 174])
				localagain = local[2:len(local)-2]
				imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0)
				imgRef = np.array(imgRef)/255
				u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
				up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
				z= min(img.shape[0],img.shape[1])
				index = min(index,img.shape[0],img.shape[1])
				z=438
				feature = np.zeros((1,z), dtype=float)
				for p in range (0,z):
					tempu =0.0
					tempv =0.0
					for q in range (0,z):
						tempu = tempu + u[p][q]*up[p][q]
						tempv = tempv + v[p][q]*vp[p][q]
					feature[0][p]= tempu + tempv
				# print(feature.shape)
				# print(GivenSubjectiveScore[0][i-1])
				if i<139:
					training.append(feature[0])
					trainingsubjectivescore.append(GivenSubjectiveScore[0][i-1 + 982 - 174])
				else:
					testing.append(feature[0])
					testingsubjectivescore.append(GivenSubjectiveScore[0][i-1 + 982 - 174])
					destortiontype.append('fastfading')
					imagename.append(localagain)
				# print(feature.shape)
		else:
			for i in range (1,174):
				img = cv2.imread(f"../Live_Database/databaserelease2/{GivenImg[start]}/img{i}.bmp", 0)
				img = np.array(img)/255
				local = np.array_str(RefImagName[0][i-1 + 227 + 233 + count*174])
				localagain = local[2:len(local)-2]
				imgRef = cv2.imread(f"../Live_Database/databaserelease2/refimgs/{localagain}", 0)
				imgRef = np.array(imgRef)/255
				u,sigma,v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
				up,sigmap,vp = np.linalg.svd(imgRef, full_matrices=True, compute_uv=True)
				z= min(img.shape[0],img.shape[1])
				index = min(index,img.shape[0],img.shape[1])
				z=438
				feature = np.zeros((1,z), dtype=float)
				for p in range (0,z):
					tempu =0.0
					tempv =0.0
					for q in range (0,z):
						tempu = tempu + u[p][q]*up[p][q]
						tempv = tempv + v[p][q]*vp[p][q]
					feature[0][p]= tempu + tempv
				# print(feature.shape)
				# print(GivenSubjectiveScore[0][i-1])
				if i<139:
					training.append(feature[0])
					trainingsubjectivescore.append(GivenSubjectiveScore[0][i-1 + 227 + 233 + count*174])
				else:
					testing.append(feature[0])
					testingsubjectivescore.append(GivenSubjectiveScore[0][i-1 + 227 + 233 + count*174])
					if count == 0:
						destortiontype.append('wn')
					else:
						destortiontype.append('gblur')
					imagename.append(localagain)
				# print(feature.shape)
				count =1
	print(start)

svr_rbf.fit(training,trainingsubjectivescore)            # fitting model with training features
predictScore = np.array(svr_rbf.predict(testing))        # model predicting on testing features
testingsubjectivescore = np.array(testingsubjectivescore)
# print(predictScore)
# print(testingsubjectivescore)
# print(destortiontype)
# print(imagename)
print(scipy.stats.pearsonr(predictScore,testingsubjectivescore))
# print(index)
import pandas as pd                                   # printing to CSV file


dataset = { 'Imagename':imagename,'Destortion':destortiontype,'Predicted_Score': predictScore, 'Actual_Score': testingsubjectivescore}

df = pd.DataFrame(dataset, columns = ['Imagename','Destortion','Predicted_Score', 'Actual_Score'])
df.to_csv('svr.csv')