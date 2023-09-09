import os
import ast
import numpy as np
import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
from datetime import datetime

from tqdm.auto import tqdm
import tensorflow as tf

def activate_list_in_dataframe(df):
    for i in df.columns:
        a = df[i].values[0]
        if type(a) == type("a"):
            if a.startswith("["):
                df[i] = df[i].apply(lambda x: ast.literal_eval(x))
    return df

def process_image(image_file):
    """Decode image at given path."""
    # Method 1: return <class 'tf.Tensor'>
    image_string = tf.io.read_file(image_file)

    # Method 2: return <class 'bytes'>
    #with open(image_file, 'rb') as f:
    #    image_string = f.read() # binary-string

    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        return 0, image_string
    except tf.errors.InvalidArgumentError:
        print('{}: Invalid JPEG data or crop window'.format(image_file))
        return 1, image_string

def make_yolo_example_from_dataframe_row(df, image_string):
    # list의 형태로 담습니다.
    if isinstance(image_string, type(tf.constant(0))):  # tensor, eagertensor 이면 꺼낸다
        image_string = [image_string.numpy()]
    else:
        image_string = [image_string]

    class_dict = {0:"people", 1:"motor", 2:"car"}
    classes_name = [class_dict[i].encode('utf8') for i in row["classes"]]
    xmin = np.asarray(row["x_min"]) / row["width"]
    ymin = np.asarray(row["y_min"]) / row["height"]
    xmax = np.asarray(row["x_max"]) / row["width"]
    ymax = np.asarray(row["y_max"]) / row["height"]
    base_name = [tf.compat.as_bytes(os.path.basename(row["img_path"]))]
    # tf.example 생성
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'filename':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
            'height':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[row['height']])),
            'width':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[row['width']])),
            'classes_id':
            tf.train.Feature(int64_list=tf.train.Int64List(value=row['classes'])),
            'classes_name':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_name)),
            'x_min':
            tf.train.Feature(float_list=tf.train.FloatList(value=xmin.tolist())),
            'y_min':
            tf.train.Feature(float_list=tf.train.FloatList(value=ymin.tolist())),
            'x_max':
            tf.train.Feature(float_list=tf.train.FloatList(value=xmax.tolist())),
            'y_max':
            tf.train.Feature(float_list=tf.train.FloatList(value=ymax.tolist())), 
            'image_raw':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=image_string))
        }))
    return example


if __name__ =="__main__":
    dtypes = ["train","val","test"]
    for dtype in dtypes:
        df = pd.read_csv(f"./data/{dtype}_detection_frame.csv")
        df = activate_list_in_dataframe(df)

        tf_record = f'./data/{dtype}_visdrone.tfrecord'

        with tf.io.TFRecordWriter(tf_record) as writer:

            counter = 0
            skipped = 0

            for idx, row in tqdm(df.iterrows()):
            
                # processes the image and parse the annotation
                error, image_string = process_image(row["img_path"])

                if not error:
                    # convert voc to `tf.Example`

                    if dtype == "train" and idx % 4 ==0:
                        # train 데이터 셋은 4 프레임마다 1개 데이터 사용
                        example = make_yolo_example_from_dataframe_row(df, image_string)
                        # write the `tf.example` message to the TFRecord files
                        writer.write(example.SerializeToString())
                    elif dtype in ["val", "test"]:
                        example = make_yolo_example_from_dataframe_row(df, image_string)
                        # write the `tf.example` message to the TFRecord files
                        writer.write(example.SerializeToString())
                    counter += 1
                else:
                    skipped += 1

        print('{} : Wrote {} images to {}'.format( datetime.now(), counter, tf_record))