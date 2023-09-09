import os, cv2
from glob import glob
import pandas as pd
import tqdm
import numpy as np
import ast
import matplotlib.pyplot as plt
import tensorflow as tf
import ast
import os, cv2




def activate_list_in_dataframe(df):
    for i in df.columns:
        a = df[i].values[0]
        if type(a) == type("a"):
            if a.startswith("["):
                df[i] = df[i].apply(lambda x: ast.literal_eval(x))
    return df




def visualize_bbox_on_img(img, bbox, class_name, color=(255, 0, 0), thickness=2):

    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min +
                                                 w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color,
                  thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name,
                                                     cv2.FONT_HERSHEY_SIMPLEX,
                                                     0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img

def visualize_bbox(imgs, bboxes, class_name):

    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    img = img[y_min:y:max, x_min:x_max]

    return img

def crop_box(imgs, boxes, pad=3):
    def get_crop_box(_img, _box, _pad):
        _box = ast.literal_eval(_box)
        x1, y1, w, h = _box
        x2,y2 = x1+w+_pad, y1+h+_pad
        x1, y1 = max(0, x1-_pad), max(0, y1-_pad)
        return _img[y1:y2, x1:x2]
    crops = [get_crop_box(img, box, pad) for (img, box) in zip(imgs, boxes)]
    return crops

def plot_ftus(crops, classname, n_cols=5, height_per_row=3):

    n_rows = int(np.ceil(len(crops)/n_cols)) 
    
    plt.figure(figsize=(18, n_rows*height_per_row)) 
    for i, crop in enumerate(crops): 
        plt.subplot(n_rows, n_cols, i+1) # subplot 생성과 각 image들의 자리 배치
        plt.imshow(crop)
        plt.title(f"{classname} #{i+1}  ––  SHAPE={crop.shape[:-1]}", fontweight="bold")
        plt.axis(False)
    plt.tight_layout()
    plt.show()

def get_sample_by_class(df, object_class, root_path, count=10, random_state=1, padding=3):

    """ get sample image """

    sample_df = df[df["class"] == object_class].sample(count, random_state=random_state)
    sample_df["img_path"] = root_path + sample_df["video_name"]+sample_df["frame_index"].apply(lambda x : "/"+str(x).zfill(7) +".jpg")
    imgs = [cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_RGB2BGR) for img_path in sample_df["img_path"].values]

    boxes = sample_df["coco"].values

    crops = crop_box(imgs, boxes, pad=padding)
    plot_ftus(crops, object_class)

def coco2xyxy(boxes):
    def _coco2xyxy(box):
        xmin,ymin,w,h = box
        xmax = xmin + w
        ymax = ymin + h
        return [xmin, ymin, xmax, ymax]
    return [_coco2xyxy(box) for box in boxes]

def coco2yolo(boxes):
    def _coco2yolo(box):
        xmin,ymin,w,h = box
        xmin = xmin + w/2
        ymin = ymin + h/2
        return [xmin, ymin, w, h]
    return [_coco2yolo(box) for box in boxes]