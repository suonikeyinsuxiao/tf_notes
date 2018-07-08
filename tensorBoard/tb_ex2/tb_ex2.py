#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: tb_ex2.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-30 15:52:27
############################

import tensorflow as tf
#将输入定义放入各自的命名空间中,从而使得TensorBoard可以根据命名空间来整理可视化效果图上的节点
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")

with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], name="add")


writer = tf.summary.FileWriter("tb_ex2.log", tf.get_default_graph())
writer.close()
