import cv2
import numpy as np
from detecter import Detecter
from threading import Thread

class VideoFileCamera(Thread): 
    def __init__(self, video_file, queue): 
        super().__init__() 
        self.queue = queue 
        self.video_file = video_file 
        self.THRESHOLD = 0.25
        self.detecter = Detecter()
    def run(self):
        cap = cv2.VideoCapture(self.video_file)
        fps = cap.get(cv2.CAP_PROP_FPS) 
        delay = int(1000/fps)
        # detecter = Detecter()
        self.detecter.setup('./frozen_inference_graph.pb', './mscoco_label_map.pbtxt') 
        while(cap.isOpened()):
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # frame_ex = np.expand_dims(frame, axis=0)
            # (boxes, scores, classes, num) = self.detecter.detect(frame_ex)
            # self.detecter.viaulize(frame, boxes, classes, scores, self.THRESHOLD)
            self.queue.put(frame, timeout=2) 
            
            cv2.waitKey(delay) 
        
        cap.release() 
        cv2.destroyAllWindows()