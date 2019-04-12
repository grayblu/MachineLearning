from vfcamera import VideoFileCamera
from imageviewer import ImageViewer
from imagesender import ImageSender

url = 'http://localhost:8080/start/camera/1'

viewer = ImageViewer('image')
sender = ImageSender(url)

camera = VideoFileCamera('OpenCV/video1.avi')
camera.addObserver(viewer)
camera.addObserver(sender)

camera.start()
viewer.start()
sender.start()