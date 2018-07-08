#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: mnist_train.py
#Brief: 神经网络的训练程序
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-30 17:01:04
############################

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py中定义的常量和前向传播的函数
import mnist_inference

#配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.3
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH = "mnist_model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    #将处理输入数据的计算都放在名字为"input"的命名空间下
    with tf.name_scope('input'):
    #定义输入输出placeholder
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-intput')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    #将处理滑动平均相关的计算都放在名为"moving_average"的命名空间下
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #将计算损失函数相关的计算都放在名为loss_function的命名空间下
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #将定义学习率,优化方法以及每一轮训练需要指定的操作都放在名字为"train_step"的命名空间下
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    #初始化TF持久化类
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter("tb_mnist.log", tf.get_default_graph())

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #在训练过程中不在测试模型在验证数据上的表现,验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            #每1000轮保存一次模型
            if i % 1000 == 0:
                #配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                #运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()
                #将配置信息和记录运行信息的proto传入运行的过程,从而记录运行时每一个节点的时间,空间开销信息
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys}, options=run_options, run_metadata=run_metadata)
                #将节点在运行时的信息写入日志文件
                writer.add_run_metadata(run_metadata, 'step%03d' % i)


                #输出当前的训练情况.这里只输出了模型在当前训练batch上的损失函数大小.通过损失函数的大小可以大概了解训练的情况.在验证数据集上的正确率信息会有一个单独的程序来完成.
                print("After {0:d} training step(s), loss on training batch is {1:e}.".format(step, loss_value))
                #保存当前模型.注意这里给出了global_step参数,这样可以让每个被保存模型的文件名末尾加上训练的轮数.比如"model.ckpt-1000"表示训练1000轮后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})


    writer.close()

def main(argv = None):
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

