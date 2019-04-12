from PIL import Image
import numpy as np
import os

target_dir = 'C:/Users/student/machineLearning/MNIST/Image/'
save_dir = 'C:/Users/student/machineLearning/MNIST/ImagetoCSV/'
list = os.listdir(target_dir)
for dir in list:
    os.makedirs(save_dir+dir)
    file_dir = target_dir + '/' + dir
    file_list = os.listdir(file_dir)
    for file in file_list:
        
        # print(file)
        image_dir = file_dir + '/' + file
        im = Image.open(image_dir)
        a = np.asarray(im)
        src, _ = os.path.splitext(file)
        # print(src)
        np.savetxt(save_dir+dir+'/'+src+'.csv', a,
                 fmt='%10.5f', delimiter=',')
