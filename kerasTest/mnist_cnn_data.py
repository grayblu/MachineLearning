from keras import datasets
from keras import backend
from keras.utils import np_utils

class Data:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        num_classes = 10
        img_rows, img_cols = x_train.shape[1:]
        
        if backend.image_data_format() == 'channels_last':
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        else:
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

def load_data():
    return Data()