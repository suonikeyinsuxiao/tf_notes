#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: saver.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-22 22:12:52
############################

"""
checkpoint                  #保存所有的模型文件列表
my_test_model.ckpt.data-00000-of-00001
my_test_model.ckpt.index
my_test_model.ckpt.meta     #保存计算图的结构信息,即神经网络的结构
"""


import tensorflow as tf

#声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

#声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    #将模型保存到指定路径
    saver.save(sess,"my_test_model.ckpt")

