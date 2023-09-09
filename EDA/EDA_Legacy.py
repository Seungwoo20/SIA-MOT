import os, cv2
from glob import glob
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import ast
import matplotlib.pyplot as plt

def make_dataset(anno_path, seq_path):
    """
    Make dataframe by annotation file

    input : 
    anno_path : annotation txt 파일 들이있는 폴더 path
    seq_path  : video folder 가 들어있는 폴더 path

    output :
    dataframe
    """

    def cls_unify(x):
        if x == 0:
            return 0
        elif x <= 2:
            return 1
        else:
            return 2

    df = pd.DataFrame(columns=[
        "video_name", "frame_index", "shape", "target_id", "coco", "norm_coco",
        "object_size", "norm_object_size", "score", "class", "truncation",
        "occlusion"
    ])

    v_name, frame, id, coco, score, cls, trun, occ, shape, object_size, norm_coco, norm_object_size = [],[],[],[],[],[],[],[],[],[],[],[]

    g_anno = glob(anno_path + "/*")
    g_seq = glob(seq_path + "/*")

    for anno, seq in tqdm(zip(g_anno, g_seq)):
        base, name = os.path.split(anno)

        name, ex = os.path.splitext(name)
        name, ex = os.path.splitext(name)
        base, _ = os.path.split(base)

        img = cv2.imread(seq + "/0000001.jpg")
        img_shape = img.shape

        with open(anno, "r") as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                line = line.split(",")
                v_name.append(name)
                frame.append(int(line[0]))
                id.append(int(line[1]))
                temp_coco = list(map(int, line[2:6]))
                coco.append(temp_coco)
                nx = temp_coco[0] / img_shape[1]
                ny = temp_coco[1] / img_shape[0]
                nw = temp_coco[2] / img_shape[1]
                nh = temp_coco[3] / img_shape[0]
                norm_coco.append([nx, ny, nw, nh])

                score.append(int(line[6]))
                cls.append(int(line[7]))
                trun.append(int(line[8]))
                occ.append(int(line[9]))
                shape.append(img_shape)
                object_size.append(temp_coco[2] * temp_coco[3])
                norm_object_size.append(nw * nh)

    df["video_name"] = v_name
    df["frame_index"] = frame
    df["shape"] = shape
    df["target_id"] = id
    df["coco"] = coco
    df["norm_coco"] = norm_coco
    df["score"] = score
    df["class"] = cls
    df["truncation"] = trun
    df["occlusion"] = occ
    df["object_size"] = object_size
    df["norm_object_size"] = norm_object_size
    df["unifed_class"] = df["class"].apply(lambda x: cls_unify(x))
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

def make_detection_dataframe(data):
    video_name = []
    frame_index = []
    heigth = []
    width = []
    coco  = []
    classes = []
    unified_classes = []

    for (k1,k2), group in data.groupby(['video_name',"frame_index"]):
        video_name.append(k1)
        frame_index.append(k2)
        heigth.append(ast.literal_eval(group["shape"].values[0])[0])
        width.append(ast.literal_eval(group["shape"].values[0])[1])
        coco.append( [ast.literal_eval(x) for x in group["norm_coco"].values] )
        classes.append( group["class"].values.tolist() )
        unified_classes.append( group["unifed_class"].values.tolist())
    df = pd.DataFrame()
    df["video_name"] = video_name
    df["frame_index"] = frame_index
    
    img_path_list = []
    for i in range(len(df)):
        img_path_list.append(df["video_name"].values[i] + "/" +str(df["frame_index"].values[i]).zfill(7) +".jpg")
    df["img_path"] = img_path_list

    df["coco"] = coco
    xyxy_list = []
    yolo_list = []
    for i in range(len(df)):
        xyxy_list.append(coco2xyxy(df["coco"].values[i]))
        yolo_list.append(coco2yolo(df["coco"].values[i]))
    df["xyxy"] = xyxy_list
    df["yolo"] = yolo_list

    df["height"] = heigth
    df["width"] = width
    df["classes"] = classes
    df["unified_classes"] = unified_classes
    return df