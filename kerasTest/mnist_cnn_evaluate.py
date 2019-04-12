from mnist_cnn_data import load_data
from dnn4 import CNN
import numpy as np
import time
from PIL import Image

im = Image.open('kerasTest/test2.png').convert("L")

im = im.resize((28,28))
im2arr = np.array(im)

print(im2arr.shape)

batch_size = 128
epochs = 10

data = load_data()
model = CNN(data.input_shape, data.num_classes)

model.load_weights('kerasTest/mnist_mlp_model3.h5')

score = model.evaluate(data.x_test, data.y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 1건 평가 시간 측정
# ix = np.random.randint(data.x_test.shape[0])

# start_time = time.time()
# result = model.predict(data.x_test[ix:ix+1]).argmax()
# end_time = time.time()
# print(result, data.y_test[ix].argmax(), end_time - start_time)

Input_data = im2arr.reshape(1, im2arr.shape[0], im2arr.shape[1], 1)

# 가공한 손글씨 확인
result = model.predict(Input_data).amax()
print(result)
