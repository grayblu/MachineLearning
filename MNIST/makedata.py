import numpy as np
import os

target_dir = 'C:/Users/student/machineLearning/MNIST/Image/'
list = os.listdir(target_dir)
for dir in list:
    work_dir = os.path.join(target_dir, dir)
    for file in os.listdir(work_dir):
        _, ext = os.path.splitext(file)
        if ext == '.ppm': pass
        np.random.shuffle(file)

