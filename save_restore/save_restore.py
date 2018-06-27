#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: save_restore.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-26 20:30:09
############################

import tensorflow as tf
from tensorflow.python.framework import graph_util

#TF提供了convert_variables_to_constants函数,通过这个函数可以将计算图中的变量及其取值通过常量的方式保存.

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图的GraphDef部分,只需要这一步就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    #将图中的变量及其取值转化为常量,同时将图中不必要的节点去掉.本程序只关心加法运算,所以这里只保存['add']节点,其他和该计算无关的节点就不保存了
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile("./combine/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())


