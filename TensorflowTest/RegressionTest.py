from LinearRegression import regression as lr
import numpy as np
import tensorflow as tf

x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
learningRate = 0.5

# w_real_column, w_real_row = x_data.shape

# print(w_real_column, w_real_row)

regression = lr(x_data,w_real,b_real,learningRate)
regression.train(10)
