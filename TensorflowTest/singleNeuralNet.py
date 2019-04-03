import tensorflow as tf
import numpy as np

# 단일 계층 신경망

x_data = [[0,0],
          [1,0],
          [1,1],
          [0,0],
          [0,0],
          [0,1]]

y_data = [[1,0,0],
          [0,1,0],
          [0,0,1],
          [1,0,0],
          [1,0,0],
          [0,0,1]]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가중치와 편향 초기화
W = tf.Variable(tf.random_uniform([2,3],-1,1))
b = tf.Variable(tf.zeros([3]))

# 신경망 구성
L = tf.add(tf.matmul(X,W),b)
# 활성함수로 ReLU
L = tf.nn.relu(L)
# 0~1사이 결과 출력을 위해 softmax함수 사용
model = tf.nn.softmax(L)

# 에러 함수(교차 엔트로피)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))

# 최적화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    # 학습을 위한 변수 초기화
    sess.run(tf.global_variables_initializer())

    # 학습
    for step in range(100):
        sess.run(train_op, feed_dict={
            X: x_data,  # 훈련 이미지
            Y: y_data   # 훈련 이미지 정답 레이블
        })
        # 중간 과정 cost 출력
        if(step+1)%10 == 0:
            print(step+1, sess.run(cost,
            feed_dict = {X: x_data, Y: y_data}))

    # 학습 결과 확인
    prediction = tf.argmax(model, axis=1)
    target = tf.argmax(Y, axis=1)

    print('예측값: ', sess.run(prediction, feed_dict={X: x_data}))
    print("실제값: ", sess.run(target, feed_dict={Y : y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
    
    print('정확도 : {:.2f}'.format(sess.run(accuracy* 100,
                     feed_dict={ X: x_data,Y: y_data})))