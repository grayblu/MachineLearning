import tensorflow as tf
import numpy as np


data  = np.loadtxt('C:/Users/student/machineLearning/Tensorflow/iris.csv', delimiter=',', dtype='float32')

np.random.shuffle(data)

train_data = data[:100]

test_data = data[100:]

print(train_data.shape, test_data.shape)

x_data = test_data[:, 1:5]
y_data = test_data[:, :1]



def one_hot(y_data):
    index = y_data.astype('uint8').flatten()
    rows = y_data.shape[0]
    cols = np.amax(index) + 1 
    temp = np.zeros((rows, cols), dtype='float32')
    temp[np.arange(rows), index] = 1.0
    return temp

def createClassifier(num_input, num_output, num_hidden1, num_hidden2):
    X = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_uniform([num_input, num_hidden1], -1, 1))
    L1 = tf.nn.relu(tf.matmul(X,W1))

    W2 = tf.Variable(tf.random_uniform([num_hidden1, num_hidden2], -1, 1))
    L2 = tf.nn.relu(tf.matmul(L1,W2))

    # 편향의 대체하는 가중치 추가
    W3 = tf.Variable(tf.random_uniform([num_hidden2, num_output], -1, 1))
    model = tf.matmul(L2, W3)

    return model


num_input = x_data.shape[1]
num_output = y_data.shape[1]
num_hidden1 = 10
num_hidden2 = 20

X2 = tf.placeholder(tf.float32)
Y2 = tf.placeholder(tf.float32)

model = createClassifier(num_input, num_output, num_output, num_hidden2)

def train(train_data, model):
    x_data = train_data[:, 1:5]
    y_data = train_data[:, :1]

    y_data = one_hot(y_data)
    
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    # 모델 저장용 변수 생성(학습 횟수를 카운트하는 변수)
    global_step = tf.Variable(0, trainable=False, name="global_step")

    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

    # 모델 저장 및 복원
    saver = tf.train.Saver(tf.global_variables())

    sess = tf.Session()

    # 학습
    feed_dict = {X: x_data, Y: y_data}
    for step in range(100):
        sess.run(train_op, feed_dict = feed_dict)
        if(step % 100 == 0){
            print('Step: {}, Cost: {}'.format(
            sess.run(global_step),
            sess.run(cost, feed_dict=feed_dict)))
        }
    # 체크포인트 저장
    saver.save(sess, './model2/dnn.ckpt', global_step=global_step)
    return saver


# 모델 저장 및 복원
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    # 체크포인트가 존재하는지 검사
    ckpt = tf.train.get_checkpoint_state('./model2')

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        # 체크포인트가 존재하는 경우, 복원
        # global_step 값도 복원됨
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # 학습
    feed_dict = {X: x_data, Y: y_data}
    for step in range(5):
        sess.run(train_op, feed_dict = feed_dict)
        print('Step: {}, Cost: {}'.format(
            sess.run(global_step),
            sess.run(cost, feed_dict=feed_dict)))
    
    # 체크포인트 저장
    saver.save(sess, './model2/dnn.ckpt', global_step=global_step)

    # 학습 결과 확인
    prediction = tf.argmax(model, axis=1)
    target = tf.argmax(Y, axis=1)

    print('예측값: ', sess.run(prediction, feed_dict={X: x_data}))
    print("실제값: ", sess.run(target, feed_dict={Y: y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
    
    print('정확도 : {:.2f}'.format(sess.run(accuracy* 100,
                     feed_dict={X: x_data, Y: y_data})))




