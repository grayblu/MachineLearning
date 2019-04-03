import numpy as np

class obj:



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
    

if __name__ == "__main__":
    # dataload 테스트
    train_data, test_data = load_data('경로', 4, 0.6)
    print(train_data.x_data)
    print(train_data.y_data)
    print(test_data.x_data)
    print(test_data.y_data)