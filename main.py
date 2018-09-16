# -*- coding:utf-8 -*-
#综合考虑,tensorlayer,keras，最终选择tensorflow keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input,InceptionV3
from tensorflow.keras.preprocessing import image
from keras.utils import np_utils
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
import numpy as np
from sklearn.datasets import load_files
from tqdm import tqdm
from PIL import ImageFile
from tensorflow.keras.models import Model
import os
import json
ImageFile.LOAD_TRUNCATED_IMAGES =True
import time
tf.logging.set_verbosity(tf.logging.INFO)
# def path_to_tensor(image_path):
#     img = image.load_img(image_path, target_size=(299,299))
#     img = image.img_to_array(img)
#     return preprocess_input(np.expand_dims(img, axis=0))
# def paths_to_tensors(images_path):
#     data = [path_to_tensor(item) for item in tqdm(images_path)]
#     return np.vstack(data)
# def load_image(images_path):
#     images = load_files(images_path)
#     onehot =  np_utils.to_categorical(images["target"],133)
#     return paths_to_tensors(images["filenames"]),onehot

def get_model():
    base_model = InceptionV3(weights='imagenet',include_top=False)
    base_model.trainable =False
    y = base_model.output
    y = GlobalAveragePooling2D()(y)
    y =  Dense(133, activation='softmax')(y)
    return Model(inputs=base_model.input,outputs=y)
if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("jobname","","ps/worker")
    tf.flags.DEFINE_integer("task_index",0,"index")
    cluster = {'master': ['10.240.208.106:2222'],
                'worker': ['10.240.208.90:2222']
	}
    jtype = FLAGS.jobname
    jindex = FLAGS.task_index
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster':cluster,
        'task':{'type':jtype, 'index':jindex}
    })
    model = get_model()
    model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])
    inputname = model.input.name.split(":")[0]
    ##将keras转换成分布氏训练
    #1.model 转换
    
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=2,cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(num_packs=2))
    runconfig = tf.estimator.RunConfig(train_distribute=distribution,session_config=tf.ConfigProto())
    estimztor_mode = tf.keras.estimator.model_to_estimator(keras_model=model,
    model_dir="/data/models",config=runconfig)
    def parse_example(ser_example):
        feats = tf.parse_single_example(ser_example,features={
            "feature":tf.VarLenFeature(tf.float32),
            "label": tf.FixedLenFeature([133], tf.float32),
            "shape": tf.FixedLenFeature([],tf.int64)
        })
        images = tf.sparse_tensor_to_dense(feats["feature"])
        images = tf.reshape(images, shape=[299,299,3])
        return images,feats["label"]

    def input_func2(image_files,batch_size=128,epoch=1):
        def myfunc():
            files =  tf.data.Dataset.list_files(image_files)
            dataset = tf.data.TFRecordDataset(files)
            dataset = dataset.repeat(epoch)
            dataset = dataset.map(parse_example)
            dataset = dataset.batch(batch_size)
            return dataset
        return myfunc
    
   # train_spec = tf.estimator.TrainSpec(input_fn=input_func2("/data/train.tfrecord",epoch=4))
#    eval_spec = tf.estimator.EvalSpec(input_fn=input_func2("/data/valid.tfrecord",epoch=4))
    estimztor_mode.train(input_fn=input_func2("/data/train.tfrecord",epoch=4))
    score = estimztor_mode.evaluate(input_fn=input_func2("/data/test.tfrecord"))
    print(score)
    print("use time:%f" %(time.time()-start_time,))
