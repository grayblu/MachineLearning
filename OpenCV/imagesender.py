from threading import Thread
import cv2
from queue import Queue
from observer import Observer
import requests

class ImageSender(Thread, Observer):
    def __init__(self, url):
        Thread.__init__(self)
        self.url = url
        self.queue = Queue(10)

    def update(self, data):
        self.queue.put(data, timeout=2)
    
    def run(self):
        while(True) :
            image = self.queue.get(timeout=2)
            ret, jpgImage = cv2.imencode('.jpg', image)
            res = requests.post(self.url, files={'image': jpgImage})