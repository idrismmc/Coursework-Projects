import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from skimage import color
img = np.array(Image.open('peppers.png'))
img=color.rgb2lab(img)
#img = img.astype('float32')


def random_center(img,k):
    centers = []
    for i in range(k):
        centers.append(random.choice(random.choice(img)))
    return centers

def find_cluster(img,centers):
    clusters = [[]for i in range(len(centers))]
    coords = [[]for i in range(len(centers))]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = find_center(centers,img[i,j])
            coord = list(np.array([i,j]))
            coords[x].append(coord)
            clusters[x].append(list(img[i,j]))
            
    return clusters,coords

def find_center(centers,pixel):
    mini = float("inf")
    for i in range(len(centers)):
        dist = np.sum(np.sqrt((centers[i]-list(pixel))**2))
        if dist<mini:
            mini = dist
            index = i
    return index

def new_center(clusters):
    centers = []
    for i in range(len(clusters)):
        centers.append(np.mean(clusters[i],axis=0))
    return centers

def convert_img5d(img):
    result = np.zeros((img.shape[0],img.shape[1],5))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i,j,0] = img[i,j,0]
            result[i,j,1] = img[i,j,1]
            result[i,j,2] = img[i,j,2]
            result[i,j,3] = i
            result[i,j,4] = j
    return result

def kmeanspp(img,k):
    centroids = []
    centroids.append(random.choice(random.choice(img)))
    for i in range(1,k):
        d2 = []
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                d2.append(min([np.sum((c-img[x,y])**2)]for c in centroids))
        probs = d2/np.sum(d2)
        cumprobs = np.cumsum(probs,axis=0)
        cumprobs = cumprobs.reshape(img.shape[0],img.shape[1])
        r = np.random.rand()
        coords = []
        for x in range(cumprobs.shape[0]):
            for y in range(cumprobs.shape[1]):
                if r<cumprobs[x,y]:
                    coords.append(x)
                    coords.append(y)
                    break
        centroids.append(img[coords[0],coords[1]])
    return centroids


def my_kmeans(inputarr,k):
#to run kmeans++ use kmeanpp() and comment random_center()

    #centroids = random_center(inputarr,k) #random centroids
    centroids = kmeanspp(inputarr,k) #kmeans++
    for i in range(5):
        print("iteration - ",i)
        clusters,coords = find_cluster(inputarr,centroids)
        centroids = new_center(clusters)
    return centroids,coords


k = 10
print("k = ",k)
#calling function to append coordinates information
kmeans_input = convert_img5d(img)

#kmeans on LAB colour format
print("kmeans using LAB colour format without coordinates")
centroids1,coords1 = my_kmeans(img,k)

output_img1 = np.zeros(img.shape)
counter = 0
for i in coords1:
    for j in i:
        output_img1[j[0],j[1]]=centroids1[counter][0:3]
    counter+=1
output_img1 = color.lab2rgb(output_img1)
output_img1 = Image.fromarray((output_img1*255).astype('uint8'))
#output_img1.save('kpp_lab_k25iter15.jpg')


#kmeans on LAB colour format with coordinates information
print("kmeans using LAB colour format with coordinates")
centroids2,coords2 = my_kmeans(kmeans_input,k)

output_img2 = np.zeros(img.shape)
counter = 0
for i in coords2:
    for j in i:
        output_img2[j[0],j[1]]=centroids2[counter][0:3]
    
    counter+=1
output_img2 = color.lab2rgb(output_img2)
output_img2 = Image.fromarray((output_img2*255).astype('uint8'))
#output_img2.save('kpp_lab5d_k25iter15.jpg')

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.axis('off')
ax2.axis('off')
ax1.set_title('LAB colour format without coordinates')
ax1.imshow(output_img1)
ax2.set_title('LAB colour format with coordinates')
ax2.imshow(output_img2)
plt.show()
