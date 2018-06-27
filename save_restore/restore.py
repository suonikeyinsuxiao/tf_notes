#!/usr/bin/env python3
#-*- coding:utf-8 -*-
############################
#File Name: restore.py
#Brief:
#Author: frank
#Mail: frank0903@aliyun.com
#Created Time:2018-06-22 22:34:16
############################

"""
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
print(v1)
result = v1 + v2
print(result)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "my_test_model.ckpt")
    print(sess.run(result))


#运行结果:
#<tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>
#Tensor("add:0", shape=(1,), dtype=float32)
#[3.]

"""

#上面的过程中还是定义了 图的结构，有点重复了，那么可不可以直接从以保存的ckpt中加载图呢?

"""
import tensorflow as tf

saver = tf.train.import_meta_graph("my_test_model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "my_test_model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
"""


"""
#上面的程序，默认保存和加载了计算图中的全部变量,但有时可能只需要保存或加载部分变量。因为并不是所有隐藏层的参数需要重新训练。
#那么，怎样保存和加载部分变量呢?
#在声明tf.tain.Saver类时可以提供一个列表来制定需要保存或加载的变量.
#例如使用saver=tf.train.Saver([v1]),那么只有变量v1会被加载.
import tensorflow as tf
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

#tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value v2
saver = tf.train.Saver([v1])

with tf.Session() as sess:
    saver.restore(sess, "my_test_model.ckpt")
    print(sess.run(result))
"""
import tensorflow as tf
#保存或加载时给变量重命名
a1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
a2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = a1 + a2

#使用字典来重命名变量就可以加载原模型中的相应变量.如下指定了原来名称为v1的变量现在加载到变量a1中,原来名称为v2的变量现在加载到变量a2中
saver = tf.train.Saver({"v1":a1, "v2":a2})
#因为有时候模型保存时的变量名称和加载时的变量名称不一致,为了解决这个问题,TF可以通过字典将模型保存时的变量名和需要加载的变量关联起来.

with tf.Session() as sess:
    saver.restore(sess, "my_test_model.ckpt")
    print(sess.run(result))
