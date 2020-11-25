# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 02:29:01 2020

@author: Neal
"""
import numpy as np
import tensorflow as tf
# from tensorflow import train
# from tensorflow.train import Checkpoint
"""Checkpoint = tf.train.Checkpoint


checkpoint.read("/tmp/checkpoint").assert_consumed()

"""


# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
"""with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/tmp/model-392409.ckpt.meta')
    saver.restore(sess, "/tmp/model-392409.ckpt")
    
    """


model_path = "model-392409.ckpt"
detection_graph = tf.Graph()
with tf.Session(graph=detection_graph) as sess:
    # Load the graph with the trained states
    loader = tf.train.import_meta_graph(model_path+'.meta')
    loader.restore(sess, model_path)
    print("Done Loading")
    a = np.fromfile('test.dat')
    with sess.as_default():
        sess.run(a)

