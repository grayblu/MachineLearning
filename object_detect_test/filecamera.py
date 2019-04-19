import cv2
import numpy as np
from detecter import Detecter
from vfcamera import VideoFileCamera
from imageviewer import ImageViewer
from queue import Queue


queue = Queue(1)
filePath = 'TrafficLight.mp4'

viewer = ImageViewer('Video', queue)
viewer.start()

camera = VideoFileCamera(filePath, queue)
camera.start()





# class FileCamera:
#     def __init__(self, file_path):
#         self.cap = cv2.VideoCapture(file_path)
#         self.THRESHOLD = 0.25

#     def detect(self):
#         queue = Queue(10)


#         # while(self.cap.isOpened()):
#         #     _, frame = self.cap.read()
#         #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         #     detecter = Detecter()
#         #     detecter.setup('./frozen_inference_graph.pb', './mscoco_label_map.pbtxt')
#         #     image_ex = np.expand_dims(frame, axis=0)
#         #     (boxes, scores, classes, num) = detecter.detect(image_ex)
#         #     detecter.viaulize(frame, boxes, classes, scores, self.THRESHOLD)

#         #     cv2.imshow('Video', cv2.resize(frame,(800,600)))
#             cv2.waitKey(1)


#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()


        

# if __name__ == "__main__":

#     camera = FileCamera('TrafficLight.mp4')
#     camera.detect()

