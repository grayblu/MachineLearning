import numpy as np
import cv2



# def image(gray, cascade):
    

#     # 얼굴 인식 실행하기
#     face_list = cascade.detectMultiScale(gray,
#                                 scaleFactor=1.1,
#                                 minNeighbors=1,
#                                 minSize=(10,10))

#     if len(face_list) > 0:
#         print(face_list)

#         # 얼굴 영역에 사각형 그리기
#         color = (0,0,255)
#         for face in face_list:
#             x,y,w,h = face
#             cv2.rectangle(gray, (x,y), (x+w, y+h), color, thickness=8)

#     return gray

cascade_file = 'C:/Users/student/Downloads/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

cap = cv2.VideoCapture('OpenCV/video2.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    car_list = cascade.detectMultiScale(gray,
                                scaleFactor=1.1,
                                minNeighbors=3,
                                minSize=(10,10))

    if len(car_list) > 0:
        print(car_list)

        # 얼굴 영역에 사각형 그리기
        color = (0,0,255)
        for car in car_list:
            x,y,w,h = car
            cv2.rectangle(gray, (x,y), (x+w, y+h), color, thickness=8)


    cv2.imshow('frame', gray)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




