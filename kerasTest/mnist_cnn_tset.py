from mnist_cnn_data import load_data
from dnn4 import CNN

batch_size = 128
epochs = 10

data = load_data()
model = CNN(data.input_shape, data.num_classes)

history = model.fit(data.x_train, data.y_train,
                    epochs=epochs,
                    validation_split=0.2)

score = model.evaluate(data.x_test, data.y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('kerasTest/mnist_mlp_model3.h5')