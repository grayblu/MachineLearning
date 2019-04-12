import numpy as np

class Obj:
    def __init__(self):
        pass

# one hot 인코더
def one_hot(y_data):
    # 2차원 -> 1차원
    index = y_data.astype('uint8').flatten()
    # y_data.shape
    rows = y_data.shape[0]
    cols = index.max() + 1
    temp = np.zeros((rows, cols), dtype='float32')
    temp[np.arange(rows),index] = 1.0
    return temp


def load_data(file_path, num_feature, train_rate):
    data = np.loadtxt(file_path, delimiter=',', dtype='float32')

    total = data.shape[0]
    base = int(total * train_rate)

    np.random.shuffle(data)
    train_data = data[:base]
    test_data = data[base:]

    train = Obj()
    test = Obj()

    train.x_data = train_data[:, 1:num_feature]
    train.y_data = one_hot(train_data[:, :1])

    test.x_data = test_data[:, 1:num_feature]
    test.y_data = one_hot(test_data[:, :1])
    
    return train, test

if __name__ == "__main__":
    # dataload 테스트
    train_data, test_data = load_data('C:/Users/student/machineLearning/TensorflowTest/iris.csv', 4, 0.6)
    print(train_data.x_data)
    print(train_data.y_data)
    print(test_data.x_data)
    print(test_data.y_data)