import tesnsorflow as tf

class Classifier:
    def __init__(self, num_input, num_output, num_hidden, num_hidden2, lr):
        
        self.X, self.Y = self.make_placeholder()

        self.model = self.make_model()

        self.prediction = tf.argmax(self.model, axis=1)

    def make_placeholder(self):
        return
    
    def make_model(self, X, num_input, num_output, num_hidden, num_hidden2):
        W1 = tf.Variable(tf.random_uniform([num_input, num_hidden], -1, 1))
        L1 = tf.nn.relu(tf.matmul(X,W1))

        W2 = tf.Variable(tf.random_uniform([num_hidden, num_hidden2], -1, 1))
        L2 = tf.nn.relu(tf.matmul(L1,W2))

        # 편향의 대체하는 가중치 추가
        W3 = tf.Variable(tf.random_uniform([num_hidden2, num_output], -1, 1))
        return tf.matmul(L2, W3)
    
    def make_train_op(self, Y, model, lr):
        cost = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(label=Y, logits=model))
    

    def make_session(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess

    # 훈련
    def train(self, data, step_num, prn_num):
        train_dict = {self.X: data.x_data, self.Y: data.y_data}

        for step in range(step_num):
            self.sess.run(self.make_train_op, feed_dict= train_dict)

            if(step+1)%prn_num == 0:
                print()
    
    # 테스트
    def test(self, data):
        print('예측값: ', self.sess.run(self.prediction, feed_dict={X: x_data}))
        
        test_dict = {self.X: data.x_data, self.Y: data.y_data}
        target = tf.argmax(self.Y, axis=1)

        print("실제값: ", sess.run(target, feed_dict={self.Y : data.y_data}))
        
        is_correct = tf.equal(self.prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
        
        print('정확도 : {:.2f}'.format(sess.run(accuracy* 100,
                        feed_dict=test_dict)))
    

    # 질의
    def query(self, x_data):
        return self.sess.run(self.prediction, feed_dict={self.X: x_data})
    
    def close(self):
        self.sess.close()