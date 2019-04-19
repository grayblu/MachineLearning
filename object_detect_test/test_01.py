import numpy as np
from PIL import Image
from detecter import Detecter
from detecter_image import get_detect_image
import cv2

# 테스트 이미지 파일 리스트
TEST_IMAGE_PATHS = [ './image1.jpg'] #, './image2.jpg']
THRESHOLD = 0.3

detecter = Detecter()
detecter.setup('./frozen_inference_graph.pb',
                 './mscoco_label_map.pbtxt')

for image_path in TEST_IMAGE_PATHS:
        image, image_ex = get_detect_image(image_path)
        (boxes, scores, classes, num) = detecter.detect(image_ex)


        object_list = filter(lambda item:item[1] > THRESHOLD, zip(boxes, scores, classes))
        (height, width, _) = image.shape
        for ix, object in enumerate(object_list):
                box = object[0]
                (sy, sx, ey, ex) = (int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width))

                sub_image = image[sy:ey, sx:ex]
                print(sub_image.shape)
        

   
cv2.waitKey()
cv2.destroyAllWindows()


    # print('object num', num)
    
    # (boxes, scores, classes) = (np.squeeze(boxes), np.squeeze(scores),
    #                             np.squeeze(classes).astype(np.uint8))

    # print('boxes', boxes)
    # print('scores', scores)
    # print('classes', classes)

    # detecter.viaulize(image, boxes, classes, scores, THRESHOLD)
    # cv2.imshow(image_path, image)