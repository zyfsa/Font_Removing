#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import faiss
import time

i = 0
j = 0
#sift = cv2.xfeatures2d.SIFT_create()

kpd = [[]]
desdlist = [[]]

###################### load features
############  database
for filename in os.listdir(r"/home/liuzhen/Font_Removing/database"):
    img1 = cv2.imread('/home/liuzhen/Font_Removing/database/' + filename, 0)  # databaseImage
    # print i
    # print filename
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    # kpd = [[]]
    # desdlist = [[]]
    # record the point and image.id
    for kd in kp1:
        kpd.extend([[kd.pt[0], kd.pt[1], i]])

    for pd in des1:
        desdlist.extend([pd.tolist()])

    i = i + 1

kpd.remove(kpd[0])
desdlist.remove(desdlist[0])
desd = np.array(desdlist)
xb = desd.astype('float32')
##########IVF + PQ
nlist = 100
m = 8        ####number of bytes per vector
n = 4
d = xb.shape[1]
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#start1 = time.clock()
index.train(xb)
#start2 = time.clock()
#print start2-start1
index.add(xb)   # 8 specifies that each sub-vector is encoded as 8 bits

###########
start = time.clock()

##################### query
kpq = [[]]
desqlist = [[]]

for filename in os.listdir(r"/home/liuzhen/Font_Removing/query"):
    img2 = cv2.imread('/home/liuzhen/Font_Removing/query/' + filename, 0)  # queryImage
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)
    # kpq = [[]]
    # desqlist = [[]]
    # record the point and  image.id
    for kq in kp2:
        kpq.extend([[kq.pt[0], kq.pt[1], j]])
        # print kpq

    for pq in des2:
        desqlist.extend([pq.tolist()])
        #        print desqlist

    j = j + 1

#print kpq
#print desqlist.shape()
kpq.remove(kpq[0])
desqlist.remove(desqlist[0])
#print desqlist
### list to array
#desd = np.array(desdlist)
desq = np.array(desqlist)

############### search
#xb = desd.astype('float32')
xq = desq.astype('float32')
#print xq.shape
#nlist = 100
#m = 8
#k = 4
#d = xb.shape[1]
#quantizer = faiss.IndexFlatL2(d)  # this remains the same

#index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#start1 = time.clock()
#index.train(xb)
# start2 = time.clock()#print start2-start1
# index.add(xb)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
#print I
#print D
#print I.shape
end_index = time.clock()
print ('Running time:%s Seconds' % (end_index - start))
#### get the information about corresponding point and imag_ID
P = [[]]
#############only show the nearest neighbor 1-NN
for n in range(I.shape[0]):
    P.extend([[kpq[n][0], kpq[n][1], kpq[n][2], kpd[I[n][0]][0], kpd[I[n][0]][1], kpd[I[n][0]][2]]])

P.remove(P[0])

#end_recordloc = time.clock()
#print ('Running time:%s Seconds'%(end_recordloc-end_index))
#print P
print 'successful!'

