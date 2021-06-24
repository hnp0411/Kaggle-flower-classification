import glob
import io
from pprint import pprint
from collections import OrderedDict

from PIL import Image
import tensorflow as tf

from preprocess import convert_tfrecord


if __name__ == "__main__":
    tf.enable_eager_execution()

    #train_files = glob.glob('data/tfrecords-*/train/*.tfrec')
    val_files = glob.glob('data/tfrecords-*/val/*.tfrec')
    #test_files = glob.glob('data/tfrecords-*/test/*.tfrec')

    val_ids, val_class, val_images = convert_tfrecord(val_files, mode='val')
#    print("Convert train tfrecods...")
#    train_ids, train_class, train_images = convert_tfrecord(train_files, mode='train')
#    print("Convert test tfrecods...")
#    test_ids, test_images = convert_tfrecord(test_files, mode='test')
   
    print('\nTest convert tfrecord')
    res = OrderedDict()
    res['1st val ID'] = val_ids[0]
    res['1st val Class'] = val_class[0]
    res['1st val Image'] = repr(Image.open(io.BytesIO(val_images[0])))
    pprint(res)
