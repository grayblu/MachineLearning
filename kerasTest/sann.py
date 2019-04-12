from keras.models import Model
from keras.layers import Input, Dense, Activation

# 분류기 모델을 클래스로 정의
class SNN2(Model):
    def __init__(self, num_input, num_output):
        
        # callable 객체를 통해 함수 호출
        x = Input(shape=(num_input,))
        output = Dense(num_output)
        softmax = Activation('softmax')

        y = softmax(output(x))

        super().__init__(x,y)

        self.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])