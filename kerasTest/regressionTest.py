#%%

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([0,1,2,3,4])
y = x * 2 + 1

model= Sequential()
model.add(Dense(1, input_shape=(1,)))

# Sequential.compile()
# 모델 구성 및 오차(손실) 함수 구성

model.compile('SGD', 'mse') # 확률적 경사하강법, 평균제곱오차

model.fit(x[:2], y[:2], epochs=1000,verbose=0)
# Sequential.fit(입력, 출력레이블 [, epochs=1000, verbose=0 ...])

weights, bias = model.layers[0].get_weights()
print(weights, bias)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:]).flatten())

#%%

num_points = 1000
x_data = np.random.normal(0.0, 0.55, (num_points))
y_data = x_data * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (num_points))

model = Sequential()
model.add(Dense(1, input_shape=(1,)))


model.compile('SGD', 'mse') # 확률적 경사하강법, 평균제곱오차

model.fit(x_data, y_data, epochs=1000,verbose=0)

weights, bias = model.layers[0].get_weights()
print(weights, bias)


# 테스트
test_indexs = np.random.choice(num_points, 10) # 10개 무작위 추출
test_x = x_data[test_indexs]
test_y = y_data[test_indexs]
print('Targets :', test_y)
print('Predictions:', model.predict(test_x).flatten())

#%%

# 데이터 준비
num_points = 2000

w_real = [0.3, 0.5, 0.1]
b_real = -0.2

x_data = np.random.randn(num_points, 3)
noise = np.random.randn(1, num_points) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise
y_data = y_data.reshape(2000, 1)

model = Sequential()
model.add(Dense(1, input_shape=(3,)))
model.compile(loss='mse', optimizer='sgd')
model.summary()

# 훈련
history = model.fit(x_data, y_data, epochs=1000, verbose=0)

# 가중치, bias 확인
weights, bias = model.layers[0].get_weights()
print(weights, bias)

# history 시각화
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
