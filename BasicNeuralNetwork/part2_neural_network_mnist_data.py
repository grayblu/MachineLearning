#%%

import numpy
from neural import neuralNetwork as ne
import matplotlib.pyplot as plt

data_file = open("mnist_dataset/mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()

all_values = data_list[1].split(',')

image_array = numpy.asfarray(all_values[1:]).reshape((28,28))

plt.imshow(image_array, cmap='Greys', interpolation='None')

#%%

#scale input to range 0.01 to 1.00
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)

#%%

# output nodes is 10 (example)
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)

#%%
from neural import neuralNetwork as ne


#신경망의 인스턴스 생성
input_nodes = 784
hidden_nodes = 100 # 과학적 근거는 없음, 입력보다는 작고 출력보다는 크게 
output_nodes = 10
# 학습률 0.3 정의
learning_rate = 0.3

n = ne(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.train_from_file("mnist_dataset/mnist_train.csv")
n.query_from_file("mnist_dataset/mnist_test_10.csv")

# n.load_weight("minst_dataset/mnist_test")
# n.save_weight("minst_dataset/mnist_test")


# # 훈련 데이터 읽기
# training_data_file = open("mnist_dataset/mnist_train_100.csv", "r")
# training_data_list = training_data_file.readlines()
# training_data_file.close()

# # 테스트 데이터 모음 내의 모든 레코드 탐색
# for record in test_data_list:
#      # 레코드를 쉼표로 구분
#      all_values = record.split(',')
#      # 입력 값의 범위와 값 조정
#      inputs = (numpy.asfarray(all_values[1:])/255.0*0.99) + 0.01
#      # 결과 값 생성(정답은 0.99, 그 외는 모두 0.01)
#      targets = numpy.zeros(output_nodes) + 0.01
#      targets[int(all_values[0])] = 0.99 
     
#      # 학습 데이터로 신경망 훈련시키기
#      n.train(inputs, targets)