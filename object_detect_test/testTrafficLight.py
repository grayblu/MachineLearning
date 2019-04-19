import numpy as np
from PIL import Image
from detecter import Detecter
from detecter_image import get_detect_image
import cv2

def detect_red_and_yellow(img, imgName, Threshold=0.01,):

    desired_dim = (30, 90) # width, height
    img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    imgName = imgName + '.jpg'
    cv2.imwrite(imgName, img_hsv)

    # lower mask (0-10)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # upper mask (170-180)
    lower_red1 = np.array([170, 70, 50])
    upper_red1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    # defining the Range of yellow color
    lower_yellow = np.array([21, 39, 64])
    upper_yellow = np.array([40, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    # red pixels' mask
    mask = mask0 + mask1 + mask2
    # Compare the percentage of red values
    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])
    if rate > Threshold:
        print('stop')
    else:
        print('go')

# 테스트 이미지 파일 리스트
TEST_IMAGE_PATHS = [ 'Traffic-Lights.jpg']
THRESHOLD = 0.3

detecter = Detecter()
detecter.setup('./frozen_inference_graph.pb',
                 './mscoco_label_map.pbtxt')

for image_path in TEST_IMAGE_PATHS:
    image, image_ex = get_detect_image(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (boxes, scores, classes, num) = detecter.detect(image_ex)
    
    detecter.viaulize(image, boxes, classes, scores, THRESHOLD)
    cv2.imshow('image', image)
    
    object_list = filter(lambda item:item[1] > THRESHOLD, zip(boxes, scores, classes))
    (height, width, _) = image.shape
    for ix, object in enumerate(object_list):
        box = object[0]
        classes = object[2] # 분류맵

        (sy, sx, ey, ex) = (int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width))

        sub_image = image[sy:ey, sx:ex]
        imgName = "object."+str(ix)
        # print(sub_image.shape)
        if(classes == 10):
                detect_red_and_yellow(sub_image, imgName)


        # cv2.imshow(imgname, sub_image)
        # cv2.waitKey(1)
        
        # cv2.imwrite("object_image.jpg", sub_image)

   
cv2.waitKey()
cv2.destroyAllWindows()

