# %%
# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    class_name : ndarray
        Detector class.

    """

    def __init__(self, tlwh, confidence, classes, track_id=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.classes = classes
        self.track_id = track_id

        
    def get_class(self):
        return self.classes
    
    def add_track_id(self, track_id=None):
        if track_id == None:
            pass
        else:
            self.track_id = track_id
        
    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    # detection 하나만 바꿔줌
    def to_xyxy(self):
        """
        top left, width, height --> x_min, y_min, x_max, y_max
        """
        ret = self.tlwh.copy()
        ret[2] = ret[0] + ret[2]
        ret[3] = ret[1] + ret[3]
        return ret
