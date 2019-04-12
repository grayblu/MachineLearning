import cv2
import requests

url = 'http://localhost:8080/start/camera/1'

cap = cv2.VideoCapture('OpenCV/video1.avi')

while(cap.isOpened()):
    # frame 포맷은 RGB
    ret, frame = cap.read()
    # 요청한 이미지는 jpg이므로 이미지변환 
    ret, jpgImage = cv2.imencode('.jpg', frame)

    # <form><input type="file" name="image"> 와 같음
    file = { 'image' : jpgImage }
    res = requests.post(url, files=file)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()