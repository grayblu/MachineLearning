from keras.models import Sequential
from keras.layers import Input, Dense, Activation

class DNN(Sequential):
    def __init__(self, num_input, num_hidden, num_output):
        super().__init__()

        self.add(Dense(num_hidden[0], activation='relu', input_shape=(num_input,)))
        self.add(Dense(num_hidden[1], activation='relu'))
        self.add(Dense(num_output, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])