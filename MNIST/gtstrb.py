from PIL import Image
import os

def center_crop(im):
    crop_size = min(im.size)

    left = (im.size[0] - crop_size)//2  # //는 몫을 연산 결과로
    top = (im.size[1] - crop_size)//2
    right = (im.size[0] + crop_size)//2
    bottom = (im.size[1] + crop_size)//2

    return im.crop((left, top, right, bottom))

def convert_image(src_file, dest_file):
    _, ext = os.path.splitext(src_file)
    if ext != '.ppm':
        return
    
    src_image = Image.open(src_file)
    dest_image = src_image.convert('L')
    dest_image = center_crop(dest_image).resize((32,32))
    dest_image.save(dest_file)



base_dir = 'C:/Users/student/Downloads/강의자료/9. 머신러닝/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/'
target_dir = 'C:/Users/student/machineLearning/MNIST/Image/'

list = os.listdir(base_dir)
for dir in list:
    dest = target_dir + dir
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    src_dir = base_dir + dir
    file_list = os.listdir(src_dir)
    for file in file_list:
        src_file = src_dir + '/' + file
        dest_file = dest + '/' + file
        convert_image(src_file, dest_file)
    
    print(src_dir, '완료')

