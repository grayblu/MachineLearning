import sim_data
# from dnn import DNN1
from dnn2 import DNN2

x_data, y_data = sim_data.load_data()

# model = DNN1(2, 10, 3)
model = DNN2(2,10,3)


model.fit(x_data,y_data,epochs=1000,verbose=0)
loss, accuracy = model.evaluate(x_data, y_data, batch_size=100)

print(loss, accuracy)