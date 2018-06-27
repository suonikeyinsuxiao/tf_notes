#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: saver_ema.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-25 21:02:23
############################
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
v2 = tf.Variable(0, dtype=tf.float32, name="v2")
for variables in tf.global_variables():
    print(variables.name)
#v:0
#v2:0


#在声明滑动平均模型后,TF会自动生成一个影子变量
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables.name)
#v:0
#v2:0
#v/ExponentialMovingAverage:0
#v2/ExponentialMovingAverage:0

print(ema.variables_to_restore())
#{'v2/ExponentialMovingAverage': <tf.Variable 'v2:0' shape=() dtype=float32_ref>, 'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}

saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(tf.assign(v2, 10))
    sess.run(maintain_averages_op)

    saver.save(sess, "moving_average.ckpt")
    print(sess.run([v, ema.average(v)]))
#[10.0, 0.099999905]


