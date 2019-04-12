#%%

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sim_data


x_data, y_data = sim_data.load_data()

# 분류기
model = Sequential()
model.add(Dense(3, input_shape=(2,), activation='softmax'))
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
                # model history에 accuracy가 쌓임
model.summary()

# 훈련
history = model.fit(x_data, y_data, epochs=100, verbose=1)

# 평가 - compile()에 지정한 metrics로 평가 리턴
loss, accuracy = model.evaluate(x_data, y_data, batch_size=100)

# 기존 방식의 평가 
targets = np.argmax(y_data, axis=1) 
predicts = np.argmax(model.predict(x_data), axis=1) 
print('Targets:', targets ) 
print('Predictions:', predicts ) 
print('accuracy : ', np.equal(targets, predicts).astype('float32').mean())

#%%

# 클래스 사용을 통한 분류

import sim_data as sim_data
from snn import SNN1

x_data, y_data = sim_data.load_data()

model = SNN1(2,3)

model.fit(x_data, y_data, epochs=100, verbose=1)

loss, accuracy = model.evaluate(x_data, y_data, batch_size=100)

print(loss, accuracy)