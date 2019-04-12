import cv2
from threading import Thread
from observer import Observerable 

# 단일 카메라인 경우
class VideoFileCamera(Observerable, Thread):
    def __inif__(self, video_file):
        Thread.__init__(self)
        Observerable.__init__(self)

        self.video_file = video_file
        
    def run(self):
        cap = cv2.VideoCapture(self.video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000/fps)
        while(cap.isOpened()):
            ret, frame = cap.read()
            # self.queue.put(frame, timeout=2) 대신 observer클래스의 메서드 사용
            self.notifyAll(frame)
            cv2.waitKey(delay)
        
        cap.release()
        cv2.destroyAllWindows()

# 단일 카메라인 경우
# class VideoFileCamera(Thread):
#     def __inif__(self, video_file, queue):
#         super().__init__()
#         self.queue = queue
#         self.video_file = video_file
        
#     def run(self):
#         cap = cv2.VideoCapture(self.video_file)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         delay = int(1000/fps)
#         while(cap.isOpened()):
#             ret, frame = cap.read()
#             # 큐가 Full이면 대기, second 동안 제거되지 않으며 예외 발생
#             self.queue.put(frame, timeout=2)
#             cv2.waitKey(delay)
        
#         cap.release()
#         cv2.destroyAllWindows()

