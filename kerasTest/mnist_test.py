from mnist_dnn_data import load_data
from dnn3 import DNN

(train_x, train_y), (test_x, test_y) = load_data()

num_input = train_x.shape[1]
num_hiddens = [100, 50]
num_output = train_y.shape[1]


model = DNN(num_input, num_hiddens, num_output)

history = model.fit(train_x, train_y, epochs=5, batch_size=100, validation_split=0.2)
# validation_split=0.2 는 데이터를 80%는 훈련, 20%에 테스트 용도로 사용함을 의미함 
# evaluate()를 통해서 train, test 결과 리턴
performance_tset = model.evaluate(train_x, train_y, batch_size=100)
print('Test Loss and Accuracy->', performance_tset)