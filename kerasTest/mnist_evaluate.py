from mnist_dnn_data import load_data
from dnn3 import DNN
import numpy as np
import time

(train_x, train_y), (test_x, test_y) = load_data()

num_input = train_x.shape[1]
num_hiddens = [100, 50]
num_output = train_y.shape[1]


model = DNN(num_input, num_hiddens, num_output)

model.load_weights('kerasTest/mnist_mlp_model2.h5')

performance_tset = model.evaluate(test_x, test_y, batch_size=100)

print('performance: ',performance_tset)




ix = np.random.randint(test_x.shape[0])

start_time = time.time()
result = model.predict(test_x[ix:ix+1]).argmax()
end_time = time.time()
print(result, end_time - start_time)