import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve


def readRGBImage(imagepath):
    image = cv.imread(imagepath)  # Height, Width, Channel
    (major, minor, _) = cv.__version__.split(".")
    if major == '3':
        image = cv2.cvtColor(imagepath, cv2.COLOR_BGR2RGB)
    else:
        # Version 2 is used, not necessary to convert
        pass
    return image

  
def saveRGBImage(imagepath, raw):
    converted_img =cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    cv2.imwrite(imagepath, converted_img)


# Processing Images - R and save Average Pixel Vales into: s_vaules variable 

count = 0
cap = cv2.VideoCapture('video_raw/Xiaomi Redmi Note 5/xiaomi_take_4.mp4')                    #Input video 
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

n = 32
dim = (n,n)
s_value=[]

pbar = tqdm(total=length)

while(count!=length):
    
    pbar.update(1)  
    
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    imageR = img.copy()
    imageR[:, :, 1:3] = 0
    
    resized = cv2.resize(imageR, dim)
    (h, w) = resized.shape[:2]
    center = (w / 2, h / 2)
    angle90 = -90
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(resized, M, (h, w))
    
    r = rotated90.flatten()
    avg_r= np.average(r)
    #print(r.shape)
    s_value.append(avg_r)
                    
    count+=1
    plt.imshow(imageR)
    
    saveRGBImage('r_channel/r_d'+str(count)+'.jpg', rotated90)
    #cv2.imshow("Hudai",rotated90)

pbar.close()


# Image resisized to 2X32x3 = 3072 values per  pixels B G R  where B,G = =0

ck=0

# Average Values of each Frame for R value 
np.savetxt("avgR_val/avg_r"+str(count)+'.csv', s_value, delimiter=",")

#load Average R values For Each Frame from csv
s_val = pd.read_csv("avgR_val/avg_r"+str(count)+'.csv', delimiter = ",")

# Plot The PPG Graph

#plt.axis([700,750,50,60])
plt.figure(figsize=(20,10))
plt.plot(s_val)
#plt.gca().invert_xaxis()
plt.xlabel('time_frames')
plt.ylabel('intensity')
print("Saving PPG Graph to graphs folder....................")
plt.savefig('graphs/ppg_'+str(count)+'.png')
#plt.show()


#correcting baseline
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


a=s_val.values.flatten()
a.shape
asymP=0.01
smoothL=500000
Corrected_base = baseline_als(a,smoothL,asymP)


plt.figure(figsize=(20,10))
plt.plot(Corrected_base)
plt.plot(s_val)

