from PIL import Image
import numpy as np
import os

def img_to_arr(file_path):
    im = Image.open(file_path)
    arr = np.array(im).flatten()
    return arr/255

target_dir = 'C:/Users/student/machineLearning/MNIST/Image/'
list = os.listdir(target_dir)
for dir in list:
    work_dir = os.path.join(target_dir,dir)
    arr_list = []
    for file in os.listdir(work_dir):
        file_path = os.path.join(work_dir, file)
        _,ext = os.path.splitext(file_path)
        if ext != '.ppm' : continue
        arr = img_to_arr(file_path)
        arr_list.append(arr)
    
    dest = os.path.join(work_dir, dir+'.csv')
    np.savetxt(dest, arr_list, fmt="%.6f", delimiter=',')
    print(dest)

