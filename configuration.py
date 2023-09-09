import pandas as pd
import albumentations as A
import cv2
dataset_name = "visdrone"
TRAIN_FROM_CHECKPOINT       = False # "saved_model/yolov3_custom"
  
# YOLO options
YOLO_TYPE                   = "yolov3" # yolov4 or yolov3
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_V3_WEIGHTS             = "./weights/yolov3.weights"
YOLO_V4_WEIGHTS             = "./weights/yolov4.weights"

# YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = False # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES           = "./visdrone.names"


TRAIN_CLASSES               = "./data/visdrone.names"
TRAIN_ANNOT_PATH            = "./data/visdrone_train_33.txt"
TEST_ANNOT_PATH             = "./data/visdrone_test.txt"
TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints/yolov3_33_aug/"
DATA_TYPE = "visdrone"

    

YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416

if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]

if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]

# Train options
TRAIN_YOLO_TINY             = False
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOGDIR                = "logs/yolov3_33_aug"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}"+"_"+dataset_name
TRAIN_LOAD_IMAGES_TO_RAM    = False # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 2
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = False
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 5
SIZE_TRAIN = ((len(pd.read_csv("./data/train_detection_frame.csv"))//3) // TRAIN_BATCH_SIZE)*TRAIN_BATCH_SIZE
                 
# TEST options
TEST_BATCH_SIZE             = 2
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.5
SIZE_TEST  = (len(pd.read_csv("./data/test_detection_frame.csv")) // TRAIN_BATCH_SIZE)*TRAIN_BATCH_SIZE
"""
#YOLOv3-TINY and YOLOv4-TINY WORKAROUND
if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32, 64]    
    YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
                               [[81,  82], [135, 169], [344, 319]],
                               [[0,    0], [0,     0], [0,     0]]]
"""

scale = 1.1
train_transforms = A.Compose(
    [
        A.Resize(TRAIN_INPUT_SIZE,TRAIN_INPUT_SIZE),
        A.PadIfNeeded(
            min_height=int(TRAIN_INPUT_SIZE * scale),
            min_width=int(TRAIN_INPUT_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=TRAIN_INPUT_SIZE, height=TRAIN_INPUT_SIZE),
        A.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=10, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.Resize(TRAIN_INPUT_SIZE,TRAIN_INPUT_SIZE),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
)