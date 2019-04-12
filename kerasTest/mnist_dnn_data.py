from keras import datasets
from keras.utils import np_utils

def load_data():
    
    (X_train,y_train), (X_test,y_test) = datasets.mnist.load_data()
    # 이미지 데이터가 cnn 입력에 맞게 되어 있음 
    # X_train, y_train의 shape (60000, 28, 28), (60000,) 
    # X_test, y_test의 shape (10000, 28, 28), (10000,)
    

    # one hot 인코딩
    y_train = np_utils.to_categorical(y_train)  
    y_test = np_utils.to_categorical(y_test)
    
    # 이미지 모양 (60000, 28, 28)을 (60000, 784)로 변경 
    L, W, H = X_train.shape 
    X_train = X_train.reshape(-1, W * H) / 255.0 
    X_test = X_test.reshape(-1, W * H) / 255.0 
    
    return (X_train, y_train), (X_test, y_test)