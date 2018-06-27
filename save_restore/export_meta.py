#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: export_meta.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-26 20:56:48
############################

import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

saver.export_meta_graph("my_test_model.ckpt.meda.json", as_text=True)

