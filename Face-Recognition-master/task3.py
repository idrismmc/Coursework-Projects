import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#function to find first 3 minimum index
def find3min(a):
    firstmin = float('inf')
    secondmin = float('inf')
    thirdmin = float('inf')
    firstin = 0
    secondin = 0
    thirdin = 0
    for i in range(len(a)):
        if a[i]<firstmin:
            thirdmin = secondmin
            secondmin=firstmin
            firstmin = a[i]
            thirdin = secondin
            secondin=firstin
            firstin = i
        elif a[i]<secondmin:
            thirdmin = secondmin
            secondmin = a[i]
            thirdin = secondin
            secondin = i
        elif a[i]<thirdmin:
            thirdmin = a[i]
            thirdin = i
    return firstin,secondin,thirdin

imgs = []
#building 2d faces matrix
for i in os.listdir("trainingset"):
    img = cv2.imread("trainingset/"+i,cv2.IMREAD_GRAYSCALE)
    img = img.flatten("F")
    imgs.append(img)
 

imgs = np.asarray(imgs).T
print(imgs.shape)

#subtracting mean from the 2d faces matrix
mean_img = np.mean(imgs,1)
mean_img = mean_img.reshape(195*231,1)
nimgs=imgs-mean_img
fig1 = plt.figure()
fig1.suptitle('Mean face image')
plt.axis('off')
plt.imshow(mean_img.reshape(195,231).T)
#fig1.savefig('meanface.jpg')

#covariance matrix aT.a
cov = nimgs.T@nimgs
cov = cov/imgs.shape[1]
values,vectors = np.linalg.eig(cov)
vectors = vectors.real
idx = values.argsort()[::-1]   
values = values[idx]
vectors = vectors[:,idx]
k = 10
kvectors = vectors[0:k]
eigenfaces = kvectors@nimgs.T
fig2, axes = plt.subplots(2,5,figsize=(15,6))
fig2.suptitle("Top-10 Eigen-faces")
counter = 0
for i in range(2):
    for j in range(5):
        axes[i][j].axis('off')
        axes[i][j].imshow(eigenfaces[counter].reshape(195,231).T)
        counter+=1

#fig2.savefig("eigenfaces.jpg")

#finding projection
weights = nimgs.T@eigenfaces.T

#loading test image
timg = cv2.imread("testset/subject10.happy.png",cv2.IMREAD_GRAYSCALE)
timg = timg.flatten("F").T

timg = timg.reshape(195*231,1)
ntimgs = timg-mean_img

tweights = ntimgs.T@eigenfaces.T
#tweights.shape

#computing 3 nearest neighbours
dist = np.linalg.norm(tweights-weights,axis=1)
min1,min2,min3 = find3min(dist)

fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(1,4,1)
ax1.axis('off')
ax2 = fig.add_subplot(1,4,2)
ax2.axis('off')
ax3 = fig.add_subplot(1,4,3)
ax3.axis('off')
ax4 = fig.add_subplot(1,4,4)
ax4.axis('off')
ax1.imshow(timg.reshape(195,231).T)
ax1.set_title('Test image')
ax2.imshow(imgs[:,min1].reshape(195,231).T)
ax2.set_title('1st closest')
ax3.imshow(imgs[:,min2].reshape(195,231).T)
ax3.set_title('2nd closest')
ax4.imshow(imgs[:,min3].reshape(195,231).T)
ax4.set_title('3rd closest')
plt.show()
#fig.savefig('subject16.2.1.png')