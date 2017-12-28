 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Make up some real data 这里是使用numpy来创建300个-1~1之间的关系
x_data = np.linspace(-1,1,300)
x_data=x_data[:, np.newaxis] #这是numpy做转置的办法
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise
plt.scatter(x_data, y_data)
plt.show()

# define placeholder for inputs to network 创建两个placeholder
xs = tf.placeholder(tf.float32,[None,1]) #创建placeholder的时候一定要指定数据的格式和数据的类型
ys = tf.placeholder(tf.float32,[None,1])

# add hidden layer 调用上面的添加隐含层
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1,10,1,activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum( tf.square(ys-prediction),reduction_indices=[1]))
optimizer  = tf.train.GradientDescentOptimizer(0.1)

train_step =  optimizer.minimize(loss)

# important step
init = tf.initialize_all_variables();

sess = tf.Session()
sess.run(init)

for i in range(1000):

    # training
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))





