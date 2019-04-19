from threading import Thread
import cv2
import numpy as np
from detecter import Detecter

class ImageViewer(Thread):
    def __init__(self, name, queue):
        super().__init__()
        self.name = name
        self.queue = queue
        self.THRESHOLD = 0.25
        self.detecter = Detecter()
        self.detecter.setup('./frozen_inference_graph.pb', './mscoco_label_map.pbtxt')
        pass
    
    def run(self):
        while(True) :
            frame = self.queue.get(timeout=2)
            frame_ex = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num) = self.detecter.detect(frame_ex)
            self.detecter.viaulize(frame, boxes, classes, scores, self.THRESHOLD)
            cv2.imshow(self.name,frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        pass