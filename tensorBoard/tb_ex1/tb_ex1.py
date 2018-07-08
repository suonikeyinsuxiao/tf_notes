#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: tb_ex1.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-30 15:52:27
############################

import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], name="add")


writer = tf.summary.FileWriter("tb_ex1.log", tf.get_default_graph())
writer.close()
