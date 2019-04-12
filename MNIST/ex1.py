import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './data'


data = input_data.read_data_sets(DATA_DIR, one_hot=True)
# 훈련 데이터
print(data.train.images.shape)
print(data.train.labels.shape)


# 입력 데이터를 위한 플레이스홀더
x = tf.placeholder(tf.float32, [None, 784])
# 가중치
W = tf.Variable(tf.zeros([784, 10]))
# 편향 - bias
b = tf.Variable(tf.zeros([10]))
# 정답 레이블을 위핚 플레이스 홀더
y_true = tf.placeholder(tf.float32, [None, 10])

# 훈련시 정답 예측값
y_pred = tf.nn.softmax(tf.matmul(x, W)+b)
# cross entropy 손실 함수 연산 노드
cross_entropy = -tf.reduce_mean(y_true * tf.log(y_pred))
# 경사하강법에 의핚 최적화
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 정답 판별 연산 노드 - 정답 예측 인덱스과 정답 레이블 인덱스 일치 여부 판단
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

# 정확도 계산 연산 노드
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))


NUM_STEPS = 1000
MINIBATCH_SIZE = 100
# 훈련 및 테스트, 정확도 측정
with tf.Session() as sess:
    # 학습을 위한 변수 초기화
    sess.run(tf.global_variables_initializer())

    # 학습
    for _ in range(100):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(train_step, feed_dict={
            x: batch_xs,        # 훈련 이미지
            y_true : batch_ys   # 훈련 이미지 정답 레이블
        })
    # 테스트
    ans = sess.run(accuracy, feed_dict = {
        x: data.test.images,
        y_true: data.test.labels
    })

    print("Accuracy: {:.4}%".format(ans*100))

# # 훈련 및 테스트, 정확도 측정
# with tf.Session() as sess:
#     # 학습을 위한 변수 초기화
#     sess.run(tf.global_variables_initializer())

#     # 학습
#     for _ in range(100):
#         sess.run(train_step, feed_dict={
#             x: data.test.images,        # 훈련 이미지
#             y_true : data.test.labels   # 훈련 이미지 정답 레이블
#         })
#     # 테스트
#     ans = sess.run(accuracy, feed_dict = {
#         x: data.test.images,
#         y_true: data.test.labels
#     })

#     print("Accuracy: {:.4}%".format(ans*100))

#%%

# 심층 신경망
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])  # 출력 0~9까지 10개

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01)) 
L2 = tf.nn.relu(tf.matmul(L1, W2)) 

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01)) 
model = tf.matmul(L2, W3) 

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y)) 
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
        _, cost_val = sess.run([optimizer, cost], feed_dict={
                            X: batch_xs, Y: batch_ys})
        
        total_cost += cost_val 
    
    print('Epoch:', '%04d' % (epoch + 1),
     'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)) 
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
print('정확도:', sess.run(accuracy, 
                feed_dict={X: mnist.test.images, 
                           Y: mnist.test.labels}))