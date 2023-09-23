# Bhailoog install IMP libraries first and CUDA 

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


imag = cv2.imread(r'C:\\Users\\user\\Downloads\\image4.jpeg')  # Change the path of the image
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
edged = cv2.Canny(bfilter, 30, 200) 
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None

for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
location

mask = np.zeros(gray.shape, np.uint8)
naya_image = cv2.drawContours(mask, [location], 0,255, -1)
naya_image = cv2.bitwise_and(imag, imag, mask=mask)
plt.imshow(cv2.cvtColor(naya_image, cv2.COLOR_BGR2RGB))
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
crop_img = gray[x1:x2+1, y1:y2+1]
plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
reader = easyocr.Reader(['en'])
result = reader.readtext(crop_img)
result

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX

res = cv2.putText(imag, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(imag, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))