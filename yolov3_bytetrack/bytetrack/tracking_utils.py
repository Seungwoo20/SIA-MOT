# %%
from onemetric.cv.utils.iou import box_iou_batch # pip install onemetric

import numpy as np


def tracks2boxes(tracks):
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
# boxs 입력 : xyxy, tracks : STrack
def match_detections_with_tracks(boxs, tracks):
    if not np.any(boxs) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, boxs)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(boxs)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

def convert_boxes(image, boxes):
    returned_boxes = []
    for box in boxes:
        box[0] = (box[0] * image.shape[1]).astype(int) 
        box[1] = (box[1] * image.shape[0]).astype(int)
        box[2] = (box[2] * image.shape[1]).astype(int)
        box[3] = (box[3] * image.shape[0]).astype(int)
        box[2] = int(box[2]-box[0])
        box[3] = int(box[3]-box[1])
        box = box.astype(int)
        box = box.tolist()
        if box != [0,0,0,0]:
            returned_boxes.append(box)
    return returned_boxes