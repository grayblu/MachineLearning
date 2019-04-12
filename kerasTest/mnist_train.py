from mnist_dnn_data import load_data
from dnn3 import DNN

(train_x, train_y), (test_x, test_y) = load_data()

num_input = train_x.shape[1]
num_hiddens = [100, 50]
num_output = train_y.shape[1]


model = DNN(num_input, num_hiddens, num_output)

history = model.fit(train_x, train_y, epochs=5, batch_size=100, validation_split=0.2)

model.save_weights('kerasTest/mnist_mlp_model2.h5')