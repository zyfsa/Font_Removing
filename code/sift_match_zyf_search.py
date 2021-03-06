#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

i = 0

for filename in os.listdir(r"D:/kwai_paper/first/chinese_digits"):
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('D:/kwai_paper/first/chinese_digits/' + filename,0)          # queryImage
    img2 = cv2.imread('D:/kwai_paper/first/test_image5.tiff',0) # trainImage

# Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks = 50)

#flann = cv2.FlannBasedMatcher(index_params, search_params)

## matches = flann.knnMatch(des1,des2,k=2)
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1,des2, k=2)

# store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #print mask.size
        if ( mask == None):
            print "not inner pot"
        else:
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            figure = plt.figure()
            i = i + 1
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

            plt.imshow(img3, 'gray'), plt.show()
            figure.savefig('D:/kwai_paper/first/save_match_image/' + filename)


    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #              singlePointColor = None,
    #               matchesMask = matchesMask, # draw only inliers
    #               flags = 2)

    #figure = plt.figure()
    #i = i + 1
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    #plt.imshow(img3, 'gray'),plt.show()
    #figure.savefig('D:/kwai_paper/first/save_match_image/'+str(i)+'match.png')