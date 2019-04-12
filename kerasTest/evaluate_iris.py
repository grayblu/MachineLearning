from dnn2 import DNN2
import dataload


_, test_data = dataload.load_data('C:/Users/student/machineLearning/kerasTest/iris.csv', 4, 0.6)

model = DNN2(3,10,3)
# 입력노드수 = train_data.x_data.shape[1] => 3
# 출력노드수 = train_data.y_data.shape[1] => 3
# 은닉층노드수 = 10


model.load_weights('kerasTest/mnist_mlp_model.h5')

loss, accuracy = model.evaluate(test_data.x_data, test_data.y_data, batch_size=100)

print(loss, accuracy)



# print('Predictions:', model.predict(test_data.x_data).flatten())

# loss, accuracy = model.evaluate(test_data.x_data, test_data.y_data, batch_size=100)

# print(loss, accuracy)
