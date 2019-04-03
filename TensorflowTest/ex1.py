#%%

import tensorflow as tf

h = tf.constant('Hello')    # 신경망의 입력 노드에 해당
w = tf.constant("world!")   # tf.constant() 상수 텐서 정의
hw = h + w
print(hw)

with tf.Session() as sess:  # tf.Session()연산 그래프 실행을 위한 인터페이스
    ans = sess.run(hw)

print(ans)

#%%

import tensorflow as tf


a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

sess = tf.Session()
# outs = sess.run(f)  # 페치(특정 연산의 결과를 꺼냄)
# sess.close()
# print('out={}'.format(outs))

fetches = [a,b,c,d,e,f]
outs = sess.run(fetches)
sess.close()
print('out={}'.format(outs))

#%%

a = tf.constant(1)
b = tf.constant(2)
c = tf.multiply(a,b)
d = tf.add(a,b)
e = tf.subtract(d,c)
f = tf.add(c,d)
g = tf.divide(f,e)

sess = tf.Session()
fetches = [a,b,c,d,e,f,g]
outs = sess.run(fetches)
sess.close()
print('out={}'.format(outs))

#%%

# c = tf.constant(4.0, name="C", dtype="float64")
# print(c)    # 동일 이름을 가진 텐서에 인덱스를 추가하여 생성됨

# 텐서의 형 변환
x = tf.constant([1,2,3], name="x", dtype=tf.float32)
print(x.dtype)
x = tf.cast(x, tf.int64)
print(x.dtype)

#%%

# 텐서 배열과 형태
c = tf.constant([[1,2,3],
                [4,5,6]])
print('List input: {}'.format(c.get_shape()))

#%%

A=tf.constant([[1,2,3],[4,5,6]])

x = tf.constant([1,0,1])
x = tf.expand_dims(x,1)
print(x.get_shape())

b = tf.matmul(A,x)

sess = tf.InteractiveSession()
print('matmul result: {}'.format(b.eval()))
sess.close()