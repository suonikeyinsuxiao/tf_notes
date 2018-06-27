#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: restore_ema.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-25 21:51:31
############################

import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")

saver = tf.train.Saver({"v/ExponentialMovingAverage":v})

with tf.Session() as sess:
    saver.restore(sess, "moving_average.ckpt")
    print(sess.run(v))
