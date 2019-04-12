from keras.models import Sequential 
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten 

class CNN(Sequential):
    def __init__(self, input_shape, num_classes): 
        super().__init__()
        
        # 3x3 필터 생성 
        self.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) 
        # 가로, 세로 방향의 공간을 줄이는 연산
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(64, (3, 3), activation='relu')) 
        self.add(MaxPooling2D(pool_size=(2, 2))) 
        self.add(Dropout(0.25)) 
        
        self.add(Flatten()) 
        
        self.add(Dense(128, activation='relu')) 
        self.add(Dropout(0.5)) 
        self.add(Dense(num_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
             optimizer='rmsprop', metrics=['accuracy'])