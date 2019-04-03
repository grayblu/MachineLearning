import tensorflow as tf
import numpy as np

class regression:
    def __init__(self, x_data, w_real, b_real,learningRate):
        self.x_data = x_data
        self.w_real = w_real
        self.b_real = b_real
        self.learningRate = learningRate



    
    def train(self, NUM_STEPS):
        self.NUM_STEPS = NUM_STEPS

        noise = np.random.randn(1,2000) * 0.1

        y_data = np.matmul(self.w_real, self.x_data.T) + self.b_real + noise

        print(y_data.shape)
        x = tf.placeholder(tf.float32, shape=[None, 3])
        y_true = tf.placeholder(tf.float32, shape=None)

        w = tf.Variable([[0,0,0]], dtype=tf.float32)
        b = tf.Variable(0, dtype=tf.float32)
        y_pred = tf.matmul(w, tf.transpose(x)) + b 


        loss = tf.reduce_mean(tf.square(y_true - y_pred))

        optimizer = tf.train.GradientDescentOptimizer(self.learningRate)
        train = optimizer.minimize(loss)

        init =tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for step in range(self.NUM_STEPS):
                sess.run(train, {x:self.x_data, y_true:y_data})
                print(step, sess.run([w,b]))

