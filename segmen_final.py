import numpy as np
import cv2
from matplotlib import pyplot as plt

#read image 
img = cv2.imread('cameraman.jpg')
#roi = img

#convert to gray 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#thresh = cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret,thresh = cv2.threshold(gray_blur,110,255,cv2.THRESH_TOZERO)
# #noise removal
kernel = np.ones((1, 1), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
     kernel, iterations=2)

#kernel = np.ones((3, 3), np.uint8)
closing = cv2.dilate(opening,kernel, iterations=2)

cv2.imshow('0_image original',img)
cv2.imshow('1_blur',gray_blur)
cv2.imshow("2_Adaptive Thresholding", thresh)
cv2.imshow("3_Morphological Opening", opening)
cv2.imshow("4_Morphological Closing", closing)

#Marker labelling
ret, markers = cv2.connectedComponents(closing)
print(markers)
markers = markers+1

markers = cv2.watershed(img,markers)
img[markers == 1] = [0, 0, 255]
img[markers == 2 ] = [255,0,0]
img[markers > 2 ] = [0,255,0]
#img[markers > 3] = [100,0,0]
cv2.imshow('segmented_image',img)

# #img[markers == 0] = [0,255,0]

cv2.waitKey(0)
cv2.destroyAllWindows()



