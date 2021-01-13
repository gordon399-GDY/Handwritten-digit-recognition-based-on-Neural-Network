#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 16:05
# @Author  : Seven
# @Site    : 
# @File    : demo.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 使用tensorflow自带的工具加载MNIST手写数字集合
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name='X_placeholder')
Y = tf.placeholder(tf.int32, [None, 10], name='Y_placeholder')

n_hidden_1 = 256  # 第1个隐层
n_hidden_2 = 256  # 第2个隐层
n_input = 784     # MNIST 数据输入(28*28*1=784)
n_classes = 10    # MNIST 总共10个手写数字类别

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='bias')
}

def multilayer_perceptron(x, weights, biases):
    # 第1个隐层，使用relu激活函数
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='fc_1')
    layer_1 = tf.nn.relu(layer_1, name='relu_1')
    # 第2个隐层，使用relu激活函数
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name='fc_2')
    layer_2 = tf.nn.relu(layer_2, name='relu_2')
    # 输出层
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='fc_3')
    return out_layer

pred = multilayer_perceptron(X, weights, biases)

learning_rate = 0.01
loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y, name='cross_entropy_loss')
loss = tf.reduce_mean(loss_all, name='avg_loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))  # 在测试集上评估
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))     # 计算准确率

init = tf.global_variables_initializer()
training_epochs = 20    # 训练总轮数
batch_size = 128        # 一批数据大小
display_step = 5       # 信息展示的频度

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./ISEHS_2019', sess.graph)

    # 训练
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历所有的batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 使用optimizer进行优化
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            # 求平均的损失
            avg_loss += l / total_batch
        # 每一步都展示信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=","%.9f"%avg_loss) 

            train_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            print("Train Accuracy: %.3f" % train_acc)
            test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            print("Test Accuracy: %.3f" % test_acc)

    print("Optimization Finished!")
    writer.close()
