"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np
'''
我们的目的是要计算出weight和biases  我们模型训练出这个
'''

# create data 通过numpy创建data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3  #这是y的部分

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

#去出残差的开方
loss = tf.reduce_mean(tf.square(y-y_data))    #为什么这里使用的仅仅是取出平均数，但是没有进行求和sum
#计算残差，我们自己定义残差

optimizer = tf.train.GradientDescentOptimizer(0.5) #创建优化器
train = optimizer.minimize(loss) #优化残差

init = tf.initialize_all_variables()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)          # Very important

#不断学习出loss
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
