from threading import Thread
import cv2
from queue import Queue
from observer import Observer

class ImageViewer(Thread, Observer):
    def __init__(self, name, queue):
        Thread.__init__(self)
        self.name = name
        self.queue = queue
        pass

    def update(self, data):
        self.queue.put(data, timeout=2)

    def run(self):
        while(True):
           image = self.queue.get(timeout=2)    
           cv2.imshow(self.name, image)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
        pass


# class ImageViewer(Thread):
#     def __init__(self, name, queue):
#         super().__init__()
#         self.name = name
#         self.queue = queue
#         pass

#     def run(self):
#         while(True):
#            image = self.queue.get(timeout=2)    
#            cv2.imshow(self.name, image)
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break
#         pass

        # pass는 실행 할 것이 아무 것도 없다는 것을 의미. 따라서 아무런 동작을 하지 않고 다음 코드를 실행
        #timeout=0이면 무한 대기 상태, 스레드 종료되지 않음
        # timeout 시간 동안 추가적으로 데이터가 공급되지 않으면 예외 발생.(정상적)