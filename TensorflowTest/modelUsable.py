import tensorflow as tf
import numpy as np

data  = np.loadtxt('C:/Users/student/machineLearning/Tensorflow/data.csv', delimiter=',', dtype='float32')

x_data = data[:, 1:3]
y_data = data[:, :1]

print(y_data)

# data2  = [1:6, 1:3]
# for i in range(6):
#     for j in range(3):
        
#         data2[i, j] = y_data[i, :]

# numpy를 통해 정답 레이블을 onehot 인코딩하는 방법
# 2차원 --> 1차원
# index = y_data.astype('uint8').flatten()
# y_data.shape
# temp = np.zeros((y_data.shape[0], 3), dtype='float32')
# temp[np.arange(y_data.shape[0]), index] = 1.0
# y_data = temp

# print(y_data)

# 함수로 정의
def one_hot(y_data):
    index = y_data.astype('uint8').flatten()
    rows = y_data.shape[0]
    cols = index.argmax() + 1
    temp = np.zeros((rows, cols), dtype='float32')
    temp[np.arange(rows), index] = 1.0
    return temp

y_data = one_hot(y_data)

num_input = x_data.shape[1]
num_output = y_data.shape[1]
num_hidden1 = 10

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

num_hidden2 = 20

W1 = tf.Variable(tf.random_uniform([num_input, num_hidden1], -1, 1))
L1 = tf.nn.relu(tf.matmul(X,W1))

W2 = tf.Variable(tf.random_uniform([num_hidden1, num_hidden2], -1, 1))
L2 = tf.nn.relu(tf.matmul(L1,W2))

# 편향의 대체하는 가중치 추가
W3 = tf.Variable(tf.random_uniform([num_hidden2, num_output], -1, 1))
model = tf.matmul(L2, W3)

# 모델 저장용 변수 생성(학습 횟수를 카운트하는 변수)
global_step = tf.Variable(0, trainable=False, name="global_step")

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

# 모델 저장 및 복원
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    # 체크포인트가 존재하는지 검사
    ckpt = tf.train.get_checkpoint_state('./model')

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        # 체크포인트가 존재하는 경우, 복원
        # global_step 값도 복원됨
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # 학습
    feed_dict = {X: x_data, Y: y_data}
    for step in range(2):
        sess.run(train_op, feed_dict = feed_dict)
        print('Step: {}, Cost: {}'.format(
            sess.run(global_step),
            sess.run(cost, feed_dict=feed_dict)))
    
    # 체크포인트 저장
    saver.save(sess, './model/dnn.ckpt', global_step=global_step)

    # 학습 결과 확인
    prediction = tf.argmax(model, axis=1)
    target = tf.argmax(Y, axis=1)

    print('예측값: ', sess.run(prediction, feed_dict={X: x_data}))
    print("실제값: ", sess.run(target, feed_dict={Y : y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
    
    print('정확도 : {:.2f}'.format(sess.run(accuracy* 100,
                     feed_dict={ X: x_data,Y: y_data})))
