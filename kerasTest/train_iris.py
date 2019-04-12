from dnn2 import DNN2
import dataload

train_data, test_data = dataload.load_data('C:/Users/student/machineLearning/kerasTest/iris.csv', 4, 0.6)

model = DNN2(3,10,3)

model.fit(train_data.x_data, train_data.y_data, epochs=100, verbose=0)
# loss, accuracy = model.evaluate(train_data.x_data, train_data.y_data, batch_size=100)

# print(loss, accuracy)

model.save_weights('kerasTest/mnist_mlp_model.h5')
