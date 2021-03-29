# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
from normalization import normalization
import matplotlib.pyplot as plt
import cv2
from vgg_KR_from_P import vgg_KR_from_P
I = Image.open('stereo2012a.jpg');

#plt.imshow(I)
#uv = plt.ginput(6) # Graphical user interface to get 6 points

#####################################################################
def calibrate(im, xyz, uv):
    #Normalizing the model coordinates
    trw,xyzn = normalization(xyz)

    #Normalizing the image coordinates
    tri,uvn = normalization(uv)
    
    #Creating the assembly matrix
    A = []
    for i in range(6):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )
    
    A = np.asarray(A) 
    U, S, V = np.linalg.svd(A)

    #
    C = V[-1, :] / V[-1, -1]
    C = C.reshape(3,4)
    
    #Denormalization step
    C = np.dot( np.dot( np.linalg.pinv(tri), C), trw)
    C = C / C[-1, -1]
    
    uv2 = np.dot( C, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
    uv2 = uv2 / uv2[2, :] 
    uv2 = uv2.T
    uv2 = uv2[:,0:2]
    err = np.sqrt(np.mean(np.sum((uv2 - uv)**2, 1))) 
    print('Mean Squared Error -',err)
    fig = plt.figure()
    fig.suptitle('Calibration output')
    plt.imshow(im)
   
    for i in uv:
        plt.scatter(i[0],i[1],marker = 'o',color='red')

    for i in uv2:
        plt.scatter(i[0],i[1],marker = 'x',color='blue')
    plt.axis('off')
    
    return C




############################################################################
def homography(u2Trans, v2Trans, uBase, vBase):
    #normalizing coordinates
    tr1,uvbnorm = normalization(np.vstack((uBase,vBase)).T)
    tr2,uvtnorm = normalization(np.vstack((u2Trans,v2Trans)).T)

    #creating the assembly matrix
    A = []
    for i in range(6):
        uo,vo = uvbnorm[i,0],uvbnorm[i,1]
        ut,vt = uvtnorm[i,0],uvtnorm[i,1]
        A.append([0, 0, 0, -ut, -vt, -1, vo * ut, vo * vt, vo])
        A.append([ut, vt, 1, 0, 0, 0, -uo * ut, -uo * vt, -uo])
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)
    L = V[-1, :] / V[-1, -1]
    H = L.reshape(3,3)
    H = np.dot( np.dot( np.linalg.pinv(tr1), H ), tr2)
    H = H / H[-1, -1]

    return H 


############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q

uv = np.load('uv1.npy')
xyz = np.load('xyz1.npy')

C = calibrate(I,xyz,uv)
K, R, t = vgg_KR_from_P(C)
print('K = ',K)
print('R = ',R)
print('t = ',t)
'''
uvtrans = np.load('uvtrans.npy')
uvbase = np.load('uvbase.npy')
uT = uv1[:,0]
vT = uv1[:,1]
uB = uv2[:,0]
vB = uv2[:,1]

homography_matrix = homography(uT,vT,uB,vB)

I1 = np.array(I1)
warped_image = cv2.warpPerspective(I1,homographymatrix,(I1.shape[1],I1.shape[0]))
plt.imshow(warped_image)

for i in uvbase:
    plt.scatter(i[0],i[1],marker='x',color='red')
plt.axis('off')
plt.show()
'''