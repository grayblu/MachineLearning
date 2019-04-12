from keras.models import Sequential
from keras.layers import Input, Dense, Activation

# 심층망 분류기 모델을 클래스로 정의
class DNN1(Sequential):
    def __init__(self, num_input, num_hidden, num_output):
        super().__init__()

        self.add(Dense(num_hidden, input_shape=(num_input,),
                        activation='relu'))
        
        self.add(Dense(num_output, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])