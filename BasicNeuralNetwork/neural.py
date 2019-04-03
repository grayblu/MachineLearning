import numpy
import scipy.special
import pandas as pd

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, path=None):
        
        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 학습률
        self.lr = learningrate
        
        if(path != None):   # 저장된 가중치 사용
            self.load_weight(path)
        else:
            self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5),
                                             (self.hnodes,self.inodes))
            self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5),
                                             (self.onodes,self.hnodes))


        # 활성화 함수로 시그모이드 함수 설정
        self.activation_function = lambda x : scipy.special.expit(x)
        

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 오차 배열
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 가중치 업데이트를 위한 활성함수
        self.who += self.lr * numpy.dot(
            (output_errors*final_outputs*(1.0-final_outputs)),
            numpy.transpose(hidden_outputs)
        )

        self.wih += self.lr * numpy.dot(
            (hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
            numpy.transpose(inputs)
        )


    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        
        return final_outputs
    

    def load(self, path):
        # 데이터 읽기
        data_frame = pd.read_csv(path, header=None)
        values = data_frame.values
        labels = values[:, 0:1]
        data = values[:, 1:]/255.0*0.99 + 0.01
        return labels, data
    
    def train_from_file(self, path):
        for label, inputs in zip(*self.load(path)):
            targets = numpy.zeros(self.onodes) + 0.01
            targets[label] = 0.99
            self.train(inputs, targets)
    
    def query_from_file(self, path):
        scorecard = []
        for label, inputs in zip(*self.load(path)):
            outputs = self.query(inputs)
            answer = numpy.argmax(outputs)
            if(label == answer):
                scorecard.append(1)
            else:
                scorecard.append(0)
        
        scorecard_array = numpy.asarray(scorecard)
        return scorecard_array.sum() / scorecard_array.size
    
    def load_weight(self, path):
        self.wih = pd.read_csv(path + '_wih.csv', header=None)
        self.who = pd.read_csv(path + '_who.csv', header=None)

    def save_weight(self, path):
        pd.DataFrame(self.wih).to_csv(path + '_wih.csv',
                                            index=False, header=None)
                           # pandas의 column과 index(row) 사용하지 않음
        
        pd.DataFrame(self.who).to_csv(path + '_who.csv',
                                            index=False, header=None)


