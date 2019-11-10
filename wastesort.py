import time

import keras
import tensorflow as tf
import sys
import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

image_path = './test_input/'

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#lists all files to be tested in test_input folder
list_images=os.listdir(image_path)

with tf.Session() as sess:


    for igs in list_images:
        start_time = time.time()
        #load the image data
        image_data=tf.gfile.FastGFile(image_path+igs, 'rb').read()
        print("读取图片从文件时间:{} S".format(time.time() - start_time))

        start_time = time.time()
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        print("softmax_tensor时间:{} S".format(time.time() - start_time))




        start_time = time.time()
        predictions = sess.run(softmax_tensor, \
                               feed_dict={'DecodeJpeg/contents:0': image_data})
        print("run时间:{} S".format(time.time() - start_time))

        start_time = time.time()
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        print( '\n'+igs+' :')
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
        print("预测时间:{} S".format(time.time() - start_time))