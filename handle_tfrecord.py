import numpy as np
from sklearn.datasets import load_files
from tqdm import tqdm
from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from keras.utils import np_utils
from tqdm  import tqdm
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES =True
IMAGE_SIZE = 299
# def path_to_tensor(image_path):
#     img = image.load_img(image_path, target_size=(299,299))
#     img = image.img_to_array(img)
#     return preprocess_input(np.expand_dims(img, axis=0))
# def paths_to_tensors(images_path):
#     data = [path_to_tensor(item) for item in tqdm(images_path)]
#     return np.vstack(data)
def load_image(images_path):
    images = load_files(images_path)
    onehot =  np_utils.to_categorical(images["target"],133)
    return images["filenames"],onehot
def get_simple_tfrecord(feature,label):
    tfrecord = {}
    feat = feature.shape
    tfrecord["feature"] =  tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.tostring()]))
    tfrecord["shape"] =  tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat)))
    tfrecord["label"] =  tf.train.Feature(float_list=tf.train.FloatList(value=list(label)))
    return tf.train.Example(features= tf.train.Features(feature=tfrecord))
def make_simple(images_path,output_file="data.tfrecord"):
    tf_writer = tf.python_io.TFRecordWriter(output_file)
    images,labels = load_image(images_path)
    for index, item in tqdm(enumerate(images)):
        img = image.load_img(item, target_size=(IMAGE_SIZE,IMAGE_SIZE))
        img = image.img_to_array(img)
        img = preprocess_input(np.expand_dims(img,axis=0))
        dataser = get_simple_tfrecord(img,labels[index]).SerializeToString()
        tf_writer.write(dataser)
    tf_writer.close()

if __name__ == '__main__':
    make_simple("dogImages/train",output_file="train.tfrecord")
    make_simple("dogImages/valid",output_file="valid.tfrecord")
    make_simple("dogImages/test",output_file="test.tfrecord")
