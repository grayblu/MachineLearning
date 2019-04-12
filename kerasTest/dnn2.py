from keras.models import Model
from keras.layers import Input, Dense, Activation

# Model 객체를 활용한 심층망 분류기 클래스 정의
class DNN2(Model):
    def __init__(self, num_input, num_hidden, num_output):
        
        x = Input(shape=(num_input, ))
        hidden = Dense(num_hidden)
        relu = Activation('relu')
        output = Dense(num_output)
        softmax = Activation('softmax')

        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x,y)

        self.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])