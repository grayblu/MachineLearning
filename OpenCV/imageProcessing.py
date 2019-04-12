#%%
import cv2
import numpy as np

img = cv2.imread('OpenCV/Koala.jpg')
r,c = img.shape[0:2]

M = cv2.getRotationMatrix2D((c/2, r/2), 90, -1)
# 1: 반시계 방향, -1: 시계방향
new_img = cv2.warpAffine(img, M, (c, r))
cv2.imshow('image', new_img)

cv2.waitKey()
cv2.destroyAllWindows()


#%%
import cv2
# 임계값 처리
img = cv2.imread('OpenCV/Koala.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th_vlaue, new_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

cv2.imshow('image', new_img)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np
img = cv2.imread('OpenCV/Koala.jpg')
ker = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
])

# ker = np.ones((3,3))

# ker = np.random.rand(3,3)

new_img = cv2.filter2D(img, -1, ker)

cv2.imshow('image', new_img)
cv2.waitKey()
cv2.destroyAllWindows()

#%%

import cv2
# blur

img = cv2.imread('OpenCV/Koala.jpg')

# new_img = cv2.GaussianBlur(img, (5,5),0)
new_img = cv2.medianBlur(img, (5,5))
cv2.imshow('image', new_img)
cv2.waitKey()
cv2.destroyAllWindows()


#%%

import cv2
import numpy as np

img = cv2.imread('OpenCV/Koala.jpg')
ker = np.ones((5,5), np.uint8)
new_img = cv2.erode(img,ker,iterations=1)

cv2.imshow('image', new_img)
cv2.waitKey()
cv2.destroyAllWindows()

#%%

# 모폴로지 - 팽창, 노이즈를 메꿀때 사용
import cv2
import numpy as np

img = cv2.imread('OpenCV/Koala.jpg')
ker = np.ones((5,5), np.uint8)
new_img = cv2.dilate(img,ker,iterations=1)

cv2.imshow('filter', new_img)
cv2.waitKey()
cv2.destroyAllWindows()

#%%

import cv2

img = cv2.imread("OpenCV/Koala.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x_edges = cv2.Sobel(gray,-1,1,0,ksize=5)
y_edges = cv2.Sobel(gray,-1,0,1,ksize=5)
cv2.imshow("xedges", x_edges)
cv2.imshow("yedges", y_edges)
cv2.waitKey()
cv2.destroyAllWindows()

#%%

import cv2

img = cv2.imread('OpenCV/Penguins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
edges = cv2.Canny(gray, 100, 200, 3)
cv2.imshow('canny_edges', edges)

cv2.waitKey()
cv2.destroyAllWindows()