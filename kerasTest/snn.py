from keras.models import Sequential
from keras.layers import Input, Dense, Activation

# 분류기 모델을 클래스로 정의
class SNN1(Sequential):
    def __init__(self, num_input, num_output):
        super().__init__()

        self.add(Dense(num_output, input_shape=(num_input,),
                        activation='softmax'))
        
        self.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])