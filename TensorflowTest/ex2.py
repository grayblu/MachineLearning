#%%

import tensorflow as tf

# 변수(Variable): 그래프에서 고정된 상태를 유지

init_val = tf.random_normal((1,5), 0, 1)
var  = tf.Variable(init_val, name='var')
print('pre run:\n{}'.format(var))

init = tf.global_variables_initializer()    # 초기화 연산
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)

print('post run\n{}'.format(post_var))


#%%

a = tf.placeholder('float')
b = tf.placeholder('float')

y = tf.multiply(a,b)

sess = tf.Session()

result = sess.run(y, feed_dict={a:3, b:3})
print(result)

#%%
import numpy as np

x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=(5,10))
    w = tf.placeholder(tf.float32, shape=(10,1))
    b = tf.fill((5,1), -1.)
    xw = tf.matmul(x,w)

    xwb = xw + b

    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s, feed_dict={x:x_data, w:w_data})

    print('outs =  {}'.format(outs))

#%%

# 선형 회귀 분석

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 테스트 데이터
num_points = 1000

x_data = np.random.normal(0.0, 0.55, (1000))
y_data = x_data * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (1000))

plt.plot(x_data, y_data, 'ro')
plt.show()

# 가중치 텐서 w, 바이어스 텐서 b(0으로 초기화된 엘리먼트를 갖는 1차원 텐서)
# 손실 함수 loss 정의
# 최적화 방법: 그래디언트 디센트(경사하강법) 사용

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# **모델**
y = w * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습
for step in range(8):
    # sess.run(train)
    # print(step, sess.run(w), sess.run(b), sess.run(loss))
    
    # fetch를 통한 결과 출력
    _, val_w, val_b, val_loss = sess.run([train, w, b, loss])
    # train에 대한 값을 무시하겠다는 의미로 _에 지정
    print(step, val_w, val_b, val_loss)


    # 산포도 그리기
    plt.plot(x_data, y_data, 'ro')

    # 직선 그리기
    # plt.plot(x_data, sess.run(w) * x_data + sess.run(b))
    plt.plot(x_data, val_w * x_data + val_b)

    # x,y 축 레이블링을 하고 각 축의 최대, 최소값 범위를 지정합니다
    plt.xlabel('x')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.ylabel('y')
    plt.show()

#%%

# 다중 회기 분석(독립변수가 여러 개 인 회귀 분석)

import tensorflow as tf
import numpy as np

# 데이터를 생성하고 결과를 시뮬레이션
x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000) * 0.1
# x_data.T에서 T는 전치를 뜻함
y_data = np.matmul(w_real, x_data.T) + b_real + noise

print(x_data.shape, y_data.shape)

wb_ = []

x = tf.placeholder(tf.float32, shape=[None, 3])
y_true = tf.placeholder(tf.float32, shape=None)

w = tf.Variable([[0,0,0]], dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
y_pred = tf.matmul(w, tf.transpose(x)) + b

loss = tf.reduce_mean(tf.square(y_true-y_pred))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

NUM_STEPS = 10

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(NUM_STEPS):
        sess.run(train, {x:x_data, y_true:y_data})
        if(step%5 == 0):
            print(step, sess.run([w,b]))
            wb_.append(sess.run([w,b]))
    
    print(10, sess.run([w,b]))

