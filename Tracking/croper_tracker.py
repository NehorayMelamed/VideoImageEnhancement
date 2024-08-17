import logging
import cv2
import os
import numpy as np
import torch

### Import Statements ###
# Ensure all necessary imports are included
from skimage.draw import polygon  # for creating the segmentation mask with 1s inside the polygon
import matplotlib.path as mpath
from skimage.measure import find_contours  # for finding contours in the segmentation mask
from Segmentation.sam import get_mask_from_bbox
from Tracking.co_tracker.co_tracker import track_points_in_video_auto
import PARAMETER
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# import cv2a
# from VideoEditor.imshow_pyqt import
def BB_convert_notation_function_wrapper(BB_convert_function, BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    if type(BBs) == list:
        if type(BBs[0]) == list:
            # list of lists -> each element is a list
            BBs_new = [BB_convert_function(BB, W_image, H_image) for BB in BBs]
        else:
            BBs_new = BB_convert_function(BBs, W_image, H_image)
    elif type(BBs) == tuple:
        BBs_new = BB_convert_function(BBs, W_image, H_image)
    elif type(BBs) == np.ndarray:
        # numpy array of size (N,4) -> each row is 4 numbers
        if len(BBs.shape) == 1:
            BBs = numpy_unsqueeze(BBs, 0)
            BBs_new = [BB_convert_function(BB.tolist(), W_image, H_image) for BB in BBs]
            BBs_new = np.array(BBs_new).squeeze(0)
        else:
            BBs_new = [BB_convert_function(BB.tolist(), W_image, H_image) for BB in BBs]
            BBs_new = np.array(BBs_new)
    elif type(BBs) == torch.tensor:
        # torch tensor of size (N,4) -> each row is 4 numbers
        if len(BBs.shape) == 1:
            BBs = BBs.unsqueeze(0)
            device = BBs.device
            BBs_new = [BB_convert_function(BB.tolist(), W_image, H_image) for BB in BBs]
            BBs_new = torch.tensor(BBs_new).to(device).squeeze(0)
        else:
            device = BBs.device
            BBs_new = [BB_convert_function(BB.tolist(), W_image, H_image) for BB in BBs]
            BBs_new = torch.tensor(BBs_new).to(device)
    return BBs_new


def BB_convert_notation_XYXY_to_XYXY_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    x1 = BB_tuple[2]
    y1 = BB_tuple[3]
    W, H = x1 - x0, y1 - y0
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0_normalized, y0_normalized, x1_normalized, y1_normalized)


def BB_convert_notation_XYXY_to_XYXY_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_to_XYXY_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYXY_to_XYWH_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    x1 = BB_tuple[2]
    y1 = BB_tuple[3]
    W, H = x1 - x0, y1 - y0
    return (x0, y0, W, H)


def BB_convert_notation_XYXY_to_XYWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_to_XYWH_tuple, BBs, W_image, H_image)
    return BBs_new


def BB_convert_notation_XYXY_to_XYWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    x1 = BB_tuple[2]
    y1 = BB_tuple[3]
    W, H = x1 - x0, y1 - y0
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    w_normalized = W / W_image
    h_normalized = H / H_image
    return (x0_normalized, y0_normalized, w_normalized, h_normalized)


def BB_convert_notation_XYXY_to_XYWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_to_XYWH_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYXY_to_XcYcWH_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[Xc,Yc,W,H]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    x1 = BB_tuple[2]
    y1 = BB_tuple[3]
    W, H = x1 - x0, y1 - y0
    return (x0 + (W / 2), y0 + (H / 2), W, H)


def BB_convert_notation_XYXY_to_XcYcWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_to_XcYcWH_tuple, BBs)
    return BBs_new


def BB_convert_notation_XYXY_to_XYWHnormalized_tuple(BB_tuple, W_image=640, H_image=640):
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    x1 = BB_tuple[2]
    y1 = BB_tuple[3]
    W = x1 - x0
    H = y1 - y0
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    w_normalized = W / W_image
    h_normalized = H / H_image
    return (x0_normalized, y0_normalized, w_normalized, h_normalized)


def BB_convert_notation_XYXY_to_XYWHnormalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_to_XYWHnormalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYXY_normalized_to_XYXY_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    x1_normalized = BB_tuple[2]
    y1_normalized = BB_tuple[3]
    w_normalized = x1_normalized - x0_normalized
    h_normalized = y1_normalized - y0_normalized
    X0 = x0_normalized * W_image
    Y0 = y0_normalized * H_image
    X1 = x1_normalized * W_image
    Y1 = y1_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    return (X0, Y0, X1, Y1)


def BB_convert_notation_XYXY_normalized_to_XYXY(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_normalized_to_XYXY_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYXY_normalized_to_XYWH_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    x1_normalized = BB_tuple[2]
    y1_normalized = BB_tuple[3]
    w_normalized = x1_normalized - x0_normalized
    h_normalized = y1_normalized - y0_normalized
    X0 = x0_normalized * W_image
    Y0 = y0_normalized * H_image
    X1 = x1_normalized * W_image
    Y1 = y1_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    return (X0, Y0, W, H)


def BB_convert_notation_XYXY_normalized_to_XYWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_normalized_to_XYWH_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYXY_normalized_to_XYWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    x1_normalized = BB_tuple[2]
    y1_normalized = BB_tuple[3]
    w_normalized = x1_normalized - x0_normalized
    h_normalized = y1_normalized - y0_normalized
    X0 = x0_normalized * W_image
    Y0 = y0_normalized * H_image
    X1 = x1_normalized * W_image
    Y1 = y1_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    return (x0_normalized, y0_normalized, w_normalized, h_normalized)


def BB_convert_notation_XYXY_normalized_to_XYWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_normalized_to_XYWH_normalized_tuple, BBs,
                                                   W_image, H_image)
    return BBs_new


def BB_convert_notation_XYXY_normalized_to_XcYcWH_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    x1_normalized = BB_tuple[2]
    y1_normalized = BB_tuple[3]
    w_normalized = x1_normalized - x0_normalized
    h_normalized = y1_normalized - y0_normalized
    X0 = x0_normalized * W_image
    Y0 = y0_normalized * H_image
    X1 = x1_normalized * W_image
    Y1 = y1_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    Xc = X0 + W / 2
    Yc = Y0 + H / 2
    return (Xc, Yc, W, H)


def BB_convert_notation_XYXY_normalized_to_XcYcWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_normalized_to_XcYcWH_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYXY_normalized_to_XcYcWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    x1_normalized = BB_tuple[2]
    y1_normalized = BB_tuple[3]
    w_normalized = x1_normalized - x0_normalized
    h_normalized = y1_normalized - y0_normalized
    X0 = x0_normalized * W_image
    Y0 = y0_normalized * H_image
    X1 = x1_normalized * W_image
    Y1 = y1_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    Xc = X0 + W / 2
    Yc = Y0 + H / 2
    Xc_normalized = Xc / W_image
    Yc_normalized = Yc / H_image
    return (Xc_normalized, Yc_normalized, w_normalized, h_normalized)


def BB_convert_notation_XYXY_normalized_to_XcYcWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYXY_normalized_to_XcYcWH_normalized_tuple, BBs,
                                                   W_image, H_image)
    return BBs_new


def BB_convert_notation_XYWH_to_XYXY_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,W,H]->[X0,Y0,X1,Y1]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    return (x0, y0, x0 + W, y0 + H)


def BB_convert_notation_XYWH_to_XYXY(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_to_XYXY_tuple, BBs)
    return BBs_new


def BB_convert_notation_XYWH_to_XYXY_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,W,H]->[X0,Y0,X1,Y1]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x1 = x0 + W
    y1 = y0 + H
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0_normalized, y0_normalized, x1_normalized, y1_normalized)


def BB_convert_notation_XYWH_to_XYXY_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_to_XYXY_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYWH_to_XYWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,W,H]->[X0,Y0,X1,Y1]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    BB_W = BB_tuple[2]
    BB_H = BB_tuple[3]
    x1 = x0 + BB_W
    y1 = y0 + BB_H
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    w_normalized = BB_W / W_image
    h_normalized = BB_H / H_image
    return (x0_normalized, y0_normalized, w_normalized, h_normalized)


def BB_convert_notation_XYWH_to_XYWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_to_XYWH_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYWH_to_XcYcWH_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,W,H]->[X0,Y0,X1,Y1]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x1 = x0 + W
    y1 = y0 + H
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    w_normalized = W / W_image
    h_normalized = H / H_image
    Xc = x0 + W / 2
    Yc = y0 + H / 2
    Xc_normalized = Xc / W_image
    Yc_normalized = Yc / H_image
    return (Xc, Yc, W, H)


def BB_convert_notation_XYWH_to_XcYcWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_to_XcYcWH_tuple, BBs, W_image, H_image)
    return BBs_new


def BB_convert_notation_XYWH_to_XcYcWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [X0,Y0,W,H]->[X0,Y0,X1,Y1]
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x1 = x0 + W
    y1 = y0 + H
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    w_normalized = W / W_image
    h_normalized = H / H_image
    Xc = x0 + W / 2
    Yc = y0 + H / 2
    Xc_normalized = Xc / W_image
    Yc_normalized = Yc / H_image
    return (Xc_normalized, Yc_normalized, w_normalized, h_normalized)


def BB_convert_notation_XYWH_to_XcYcWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_to_XcYcWH_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYWH_normalized_to_XYXY_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    x0 = x0_normalized * W_image
    y0 = y0_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    return (x0, y0, W, H)


def BB_convert_notation_XYWH_normalized_to_XYXY(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_normalized_to_XYXY_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYWH_normalized_to_XYXY_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    x0 = x0_normalized * W_image
    y0 = y0_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x1 = x0 + W
    y1 = y0 + H
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0_normalized, y0_normalized, x1_normalized, y1_normalized)


def BB_convert_notation_XYWH_normalized_to_XYXY_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_normalized_to_XYXY_normalized_tuple, BBs,
                                                   W_image, H_image)
    return BBs_new


def BB_convert_notation_XYWH_normalized_to_XYWH_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    x0 = x0_normalized * W_image
    y0 = y0_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x1 = x0 + W
    y1 = y0 + H
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0, y0, W, H)


def BB_convert_notation_XYWH_normalized_to_XYWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_normalized_to_XYWH_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYWH_normalized_to_XcYcWH_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    x0 = x0_normalized * W_image
    y0 = y0_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x1 = x0 + W
    y1 = y0 + H
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    Xc = x0 + W / 2
    Yc = y0 + H / 2
    Xc_normalized = Xc / W_image
    Yc_normalized = Yc / H_image
    return (Xc, Yc, W, H)


def BB_convert_notation_XYWH_normalized_to_XcYcWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_normalized_to_XcYcWH_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XYWH_normalized_to_XcYcWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    x0_normalized = BB_tuple[0]
    y0_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    x0 = x0_normalized * W_image
    y0 = y0_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x1 = x0 + W
    y1 = y0 + H
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    Xc = x0 + W / 2
    Yc = y0 + H / 2
    Xc_normalized = Xc / W_image
    Yc_normalized = Yc / H_image
    return (Xc_normalized, Yc_normalized, w_normalized, h_normalized)


def BB_convert_notation_XYWH_normalized_to_XcYcWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XYWH_normalized_to_XcYcWH_normalized_tuple, BBs,
                                                   W_image, H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_to_XYXY_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc = BB_tuple[0]
    Yc = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    return (x0, y0, x1, y1)


def BB_convert_notation_XcYcWH_to_XYXY(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_to_XYXY_tuple, BBs)
    return BBs_new


def BB_convert_notation_XcYcWH_to_XYXY_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc = BB_tuple[0]
    Yc = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0_normalized, y0_normalized, x1_normalized, y1_normalized)


def BB_convert_notation_XcYcWH_to_XYXY_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_to_XYXY_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_to_XYWH_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc = BB_tuple[0]
    Yc = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    # x0_normalized = x0 / W_image
    # y0_normalized = y0 / H_image
    # x1_normalized = x1 / W_image
    # y1_normalized = y1 / H_image
    return (x0, y0, W, H)


def BB_convert_notation_XcYcWH_to_XYWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_to_XYWH_tuple, BBs)
    return BBs_new


def BB_convert_notation_XcYcWH_to_XYWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc = BB_tuple[0]
    Yc = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    w_normalized = W / W_image
    h_normalized = H / H_image
    return (x0_normalized, y0_normalized, w_normalized, h_normalized)


def BB_convert_notation_XcYcWH_to_XYWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_to_XYWH_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_to_XcYcWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc = BB_tuple[0]
    Yc = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    w_normalized = W / W_image
    h_normalized = H / H_image
    Xc_normalized = Xc / W_image
    Yc_normalized = Yc / H_image
    return (Xc_normalized, Yc_normalized, w_normalized, h_normalized)


def BB_convert_notation_XcYcWH_to_XcYcWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_to_XcYcWH_normalized_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_normalized_to_XYXY_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc_normalized = BB_tuple[0]
    Yc_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    Xc = Xc_normalized * W_image
    Yc = Yc_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0, y0, x1, y1)


def BB_convert_notation_XcYcWH_normalized_to_XYXY(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_normalized_to_XYXY_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_normalized_to_XYXY_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc_normalized = BB_tuple[0]
    Yc_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    Xc = Xc_normalized * W_image
    Yc = Yc_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0_normalized, y0_normalized, x1_normalized, y1_normalized)


def BB_convert_notation_XcYcWH_normalized_to_XYXY_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_normalized_to_XYXY_normalized_tuple, BBs,
                                                   W_image, H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_normalized_to_XYWH_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc_normalized = BB_tuple[0]
    Yc_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    Xc = Xc_normalized * W_image
    Yc = Yc_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0, y0, W, H)


def BB_convert_notation_XcYcWH_normalized_to_XYWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_normalized_to_XYWH_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_normalized_to_XYWH_normalized_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc_normalized = BB_tuple[0]
    Yc_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    Xc = Xc_normalized * W_image
    Yc = Yc_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (x0_normalized, y0_normalized, w_normalized, h_normalized)


def BB_convert_notation_XcYcWH_normalized_to_XYWH_normalized(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_normalized_to_XYWH_normalized_tuple, BBs,
                                                   W_image, H_image)
    return BBs_new


def BB_convert_notation_XcYcWH_normalized_to_XcYcWH_tuple(BB_tuple, W_image=640, H_image=640):
    # [Xc,Yc,W,H]->[X0,Y0,X1,Y1]
    Xc_normalized = BB_tuple[0]
    Yc_normalized = BB_tuple[1]
    w_normalized = BB_tuple[2]
    h_normalized = BB_tuple[3]
    Xc = Xc_normalized * W_image
    Yc = Yc_normalized * H_image
    W = w_normalized * W_image
    H = h_normalized * H_image
    x0 = Xc - W / 2
    y0 = Yc - H / 2
    x1 = Xc + W / 2
    y1 = Yc + H / 2
    x0_normalized = x0 / W_image
    y0_normalized = y0 / H_image
    x1_normalized = x1 / W_image
    y1_normalized = y1 / H_image
    return (Xc, Yc, W, H)


def BB_convert_notation_XcYcWH_normalized_to_XcYcWH(BBs, W_image=640, H_image=640):
    # [X0,Y0,X1,Y1]->[X0,Y0,W,H]
    BBs_new = BB_convert_notation_function_wrapper(BB_convert_notation_XcYcWH_normalized_to_XcYcWH_tuple, BBs, W_image,
                                                   H_image)
    return BBs_new


def numpy_unsqueeze(input_tensor, dim=-1):
    return np.expand_dims(input_tensor, dim)

class Tracker:
    @staticmethod
    def test_align_crops_from_BB():
        ### Load Frames from Movie or Image Folder: ###
        movie_path = r'C:\Users\dudyk\Documents\RDND_dudy\SHABAK/video_deblur_2.mp4'
        frames = []  # List to store frames
        if os.path.isdir(movie_path):  # Check if movie_path is a directory
            image_files = sorted(os.listdir(movie_path))  # List image files in the directory
            for image_file in image_files:  # Loop through each image file
                img = cv2.imread(os.path.join(movie_path, image_file))  # Read image
                frames.append(img)  # Append image to frames list
        else:  # If movie_path is a file
            cap = cv2.VideoCapture(movie_path)  # Open video file
            while cap.isOpened():  # Loop through each frame in the video
                ret, frame = cap.read()  # Read frame
                if not ret:  # Break if no frame is read
                    break
                frames.append(frame)  # Append frame to frames list
            cap.release()  # Release video capture

        ### Get reference frame: ###
        reference_frame = frames[0]  # Get the first frame
        H, W = reference_frame.shape[0:2]

        ### Get BB: ###
        initial_bbox_XYXY = Tracker.draw_bounding_box(reference_frame)  # Draw initial bounding box
        initial_bbox_XYWH = BB_convert_notation_XYXY_to_XYWH(initial_bbox_XYXY)  # Convert initial bounding box
        initial_BB_XYWH_normalized = BB_convert_notation_XYWH_to_XcYcWH_normalized(initial_bbox_XYWH, W, H)  # Convert initial bounding box to normalized coordinates
        X0,Y0,X1,Y1 = initial_bbox_XYXY
        # plt.imshow(reference_frame[Y0:Y1, X0:X1])  # Display the initial bounding box on
        aligned_crops = Tracker.align_crops_from_BB(frames,
                                                 initial_BB_XYWH_normalized,
                                                 tracking_method='opencv')  # Align crops from initial bounding box
        # plt.figure(); plt.imshow(aligned_crops[-1])  # Display the first aligned crop

        return aligned_crops

    @staticmethod
    def align_crops_from_BB(frames,
                            initial_BB_XYWH,
                            tracking_method='co_tracker',  # 'co_tracker' or 'opencv'
                            ):

        ### Get first frame: ###
        reference_frame = frames[0].copy()  # Copy the first trimmed reference_frame
        height, width = reference_frame.shape[:2]  # Get height and width of the reference_frame
        # logging.debug(f"Processing normalized box: {initial_BB_XYWH_normalized}")  # Log the normalized bounding box

        ### Convert Normalized BBox to Pixel Coordinates: ###
        # cx, cy, w, h = initial_BB_XcYcWH  # Unpack the normalized bounding box
        # x = int((cx - w / 2) )  # Calculate x-coordinate in pixels
        # y = int((cy - h / 2) )  # Calculate y-coordinate in pixels
        # w = int(w)  # Calculate width in pixels
        # h = int(h)  # Calculate height in pixels
        # initial_BB_XYWH = (x, y, w, h)  # Form pixel bounding box tuple
        # # initial_BB_XYWH = BB_convert_notation_XcYcWH_to_XYWH(initial_BB_XcYcWH, width, height)  # Convert normalized bounding box to pixel coordinates
        initial_BB_XYXY = BB_convert_notation_XYWH_to_XYXY(initial_BB_XYWH)
        # logging.debug(f"Pixel bbox: {initial_BB_XYXY}")  # Log the pixel bounding box
        # plt.figure(); plt.imshow(frames[0][y:y+h, x:x+w])  # Display the initial bounding
        # cv2.namedWindow('initial_BB')
        # cv2.imshow('initial_BB', frames[0][y:y+h, x:x+w])
        # X0,Y0,X1,Y1 = initial_BB_XYXY
        # frame_crop = frames[0][Y0:Y1, X0:X1]
        # initial_frame = frames[0]
        # # display_media(initial_frame)
        # display_media(frame_crop)

        ### Generating Mask Using SAM (Segment Anything): ###
        logging.debug("Generating mask...")  # Log mask generation start
        mask = get_mask_from_bbox(reference_frame, PARAMETER.SAM_CHECKPOINTS, bbox=initial_BB_XYXY)  # Generate mask from bounding box
        logging.debug("Mask generated successfully")  # Log mask generation success
        # display_media(BW2RGB(mask.astype(np.uint8)*255))
        # mask = None

        ### Pick tracking method and track: ###
        if tracking_method == 'co_tracker':
            aligned_crops = Tracker.align_crops_from_BB_and_mask_co_tracker(frames, initial_BB_XYXY, mask)
        elif tracking_method == 'opencv':
            aligned_crops = Tracker.align_crops_from_BB_opencv(frames, initial_BB_XYXY)
        # display_media(np.ascontiguousarray(aligned_crops[-1]))
        return aligned_crops

    @staticmethod
    def align_crops_from_BB_opencv(frames, initial_bbox_XYXY):
        ### Track Object Using OpenCV Tracker: ###
        X0,Y0,X1,Y1 = initial_bbox_XYXY  # Unpack initial bounding box coordinates
        W = X1-X0
        H = Y1-Y0
        initial_bbox_XYWH = (X0, Y0, W, H)  # Form initial bounding box
        bounding_boxes_array = Tracker.track_object_using_opencv_tracker(initial_bbox_XYWH, frames)  # Generate bounding boxes for each frame
        bounding_boxes_list = bounding_boxes_array.tolist()

        ### Align Crops and Calculate Averaged Crop: ###
        input_dict = {}
        aligned_crops = Tracker.align_crops_in_frames_using_given_bounding_boxes(input_dict, frames, user_input=bounding_boxes_list, input_method='BB')  # Align crops

        return aligned_crops

    @staticmethod
    def user_input_to_all_input_types(user_input, input_method='BB', input_shape=None):
        """
        Convert user input to all input types (bounding box, polygon, segmentation mask, and grid points).

        Args:
            user_input: User input which can be a bounding box, polygon, or segmentation mask. Can be a single input or a list of inputs.
            input_method (str, optional): Method to determine the type of user input ('BB', 'polygon', 'segmentation').
            input_shape (tuple, optional): Shape of the input (H, W).

        Returns:
            BB_XYXY: Bounding box coordinates.
            polygon_points: List of polygon points.
            segmentation_mask: Segmentation mask.
            grid_points: Grid points generated within the input region.
            flag_no_input: Flag to indicate if there was no input.
            flag_list: Flag to indicate if the input was a list.
        """
        H, W = input_shape  # Unpack input shape

        def process_single_input(single_input):
            if input_method == 'BB':
                BB_XYXY = single_input
                polygon_points, segmentation_mask = Tracker.bounding_box_to_polygon_and_mask(BB_XYXY, (H, W))
                grid_points = Tracker.generate_points_in_BB(BB_XYXY, grid_size=5)
            elif input_method == 'polygon':
                polygon_points = single_input
                BB_XYXY, segmentation_mask = Tracker.polygon_to_bounding_box_and_mask(polygon_points, (H, W))
                grid_points = Tracker.generate_points_in_polygon(polygon_points, grid_size=5)
            elif input_method == 'segmentation':
                segmentation_mask = single_input
                BB_XYXY, polygon_points = Tracker.mask_to_bounding_box_and_polygon(segmentation_mask)
                grid_points = Tracker.generate_points_in_segmentation_mask(segmentation_mask, grid_size=5)
            return BB_XYXY, polygon_points, segmentation_mask, grid_points

        if isinstance(user_input, list):
            flag_list = True
            results = [process_single_input(ui) for ui in user_input]
            BB_XYXY, polygon_points, segmentation_mask, grid_points = zip(*results)
            BB_XYXY = list(BB_XYXY)
            polygon_points = list(polygon_points)
            segmentation_mask = list(segmentation_mask)
            grid_points = list(grid_points)
            flag_no_input = [False] * len(user_input)
        else:
            flag_list = False
            if user_input is not None:
                BB_XYXY, polygon_points, segmentation_mask, grid_points = process_single_input(user_input)
                flag_no_input = False
            else:
                BB_XYXY = [0, 0, W, H]  # Default bounding box is full frame
                segmentation_mask = np.ones((H, W), dtype=np.uint8)
                polygon_points = [(0, 0), (W, 0), (W, H), (0, H)]  # Default polygon is full frame
                grid_points = Tracker.generate_points_in_BB(BB_XYXY, grid_size=5)  # Default grid points are evenly spaced in the bounding box
                flag_no_input = True

        return BB_XYXY, polygon_points, segmentation_mask, grid_points, flag_no_input, flag_list

    @staticmethod
    def align_crops_in_frames_using_given_bounding_boxes(input_dict: dict,
                                                         frames: list = None,
                                                         user_input: dict = None,
                                                         input_method: str = None) -> dict:
        """
        Align crops using user input and compute the averaged crop using affine transformation.

        Args:
            input_dict (dict): Dictionary containing the following keys:
                - 'frames' (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
                - 'user_input' (dict): User input data.
                - 'input_method' (str): Method to determine the type of user input ('BB', 'polygon', 'segmentation', 'points').
            frames (list, optional): List of frames to align. Takes precedence over input_dict if provided.
            user_input (dict, optional): User input data. Takes precedence over input_dict if provided.
            input_method (str, optional): Method to determine the type of user input. Takes precedence over input_dict if provided.

        Returns:
            dict: Dictionary containing the following keys:
                - 'aligned_crops' (list): List of aligned crops.
                - 'averaged_crop' (np.ndarray): Averaged aligned crop.
                - 'homography_matrix_list' (list): List of homography matrices.
        """
        ### Extract Inputs: ###
        frames = frames if frames is not None else input_dict.get('frames')  # Extract list of frames
        user_input = user_input if user_input is not None else input_dict.get('user_input')  # Extract user input
        input_method = input_method if input_method is not None else input_dict.get('input_method')  # Extract input method

        ### Get Frame Dimensions: ###
        h, w = frames[0].shape[:2]  # Get height and width from the first frame

        ### Convert User Input to All Input Types: ###
        BB_XYXY, polygon_points, segmentation_mask, grid_points, flag_no_input, flag_list = Tracker.user_input_to_all_input_types(
            user_input, input_method=input_method, input_shape=(h, w))

        ### Calculate Reference Crop Size and Coordinates: ###
        ref_x0, ref_y0, ref_x1, ref_y1 = BB_XYXY[0]  # Extract reference bounding box coordinates

        ### Get Coords: ###
        ref_bbox_coords = np.array([[ref_x0, ref_y0], [ref_x1, ref_y0], [ref_x0, ref_y1], [ref_x1, ref_y1]], dtype=np.float32)  # Top-left, top-right, bottom-left, bottom-right
        ref_width = ref_x1 - ref_x0  # Calculate reference bounding box width
        ref_height = ref_y1 - ref_y0  # Calculate reference bounding box height

        ### Looping Over Bounding Boxes: ###
        aligned_crops = []
        homography_matrix_list = []
        for i, current_bbox in enumerate(BB_XYXY):  # Loop through each bounding box
            ### Get Bounding Box: ###
            x0, y0, x1, y1 = current_bbox  # Extract bounding box coordinates
            x0 = max(x0, 0)  # Make sure x0 is non-negative
            y0 = max(y0, 0)  # Make sure y0 is non-negative
            x1 = min(x1, frames[i].shape[1])  # Make sure x1 is within frame width
            y1 = min(y1, frames[i].shape[0])  # Make sure y1 is within frame height

            ### Get Coords: ###
            curr_bbox_coords = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float32)  # Top-left, top-right, bottom-left, bottom-right

            ### Calculate Affine Transformation Matrix: ###
            H = cv2.getPerspectiveTransform(curr_bbox_coords, ref_bbox_coords)  # Compute perspective transformation matrix
            homography_matrix_list.append(H)  # Add homography matrix to list

            ### Apply Resize: ###
            aligned_crop = cv2.resize(frames[i][y0:y1, x0:x1], (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)  # Resize the crop
            aligned_crops.append(aligned_crop)  # Add aligned crop to list

        ### This Is The Code Block: ###
        aligned_crops_np = np.array(aligned_crops)  # Convert list of aligned crops to numpy array
        averaged_crop = np.mean(aligned_crops_np, axis=0).astype(np.uint8)  # Compute the averaged crop

        ### Prepare Output Dictionary: ###
        output_dict = {
            'aligned_crops': aligned_crops,  # List of aligned crops
            'averaged_crop': averaged_crop,  # Averaged aligned crop
            'homography_matrix_list': homography_matrix_list  # List of homography matrices
        }
        return aligned_crops_np  # Return output dictionary

    @staticmethod
    def track_object_using_opencv_tracker(initial_bbox_XYWH, frames, tracker_type='CSRT'):
        """
        Generate bounding boxes for each frame based on the initial bounding box using an OpenCV tracker.

        Args:
            initial_bbox (tuple): The initial bounding box coordinates (x0, y0, w, h).
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            tracker_type (str): Type of OpenCV tracker to use. Options are 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'.

        Returns:
            np.ndarray: Bounding boxes for each frame. Shape is [T, 4].
        """

        bounding_boxes = []  # List to store bounding boxes
        tracker = Tracker.create_tracker(tracker_type)  # Create tracker

        # initial_bbox_cv = (initial_bbox[0], initial_bbox[1], initial_bbox[2] - initial_bbox[0], initial_bbox[3] - initial_bbox[1])  # Convert to OpenCV format (x, y, w, h)
        tracker.init(frames[0], initial_bbox_XYWH)  # Initialize tracker with the first frame and initial bounding box

        ### Looping Over Frames: ###
        for frame in frames:  # Loop through each frame
            success, bbox_cv = tracker.update(frame)  # Update tracker and get new bounding box
            if success:
                bbox = (int(bbox_cv[0]), int(bbox_cv[1]), int(bbox_cv[0] + bbox_cv[2]), int(bbox_cv[1] + bbox_cv[3]))  # Convert to (x0, y0, x1, y1)
            else:
                bbox = bounding_boxes[-1]  # If tracking fails, use the previous bounding box
            bounding_boxes.append(bbox)  # Append bounding box to list

        return np.array(bounding_boxes)  # Return bounding boxes for each frame

    ### Function: create_tracker ###
    @staticmethod
    def create_tracker(tracker_type):
        """
        Create an OpenCV tracker based on the specified type.

        Args:
            tracker_type (str): Type of OpenCV tracker to use. Options are 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'.

        Returns:
            tracker: OpenCV tracker object.
        """

        if tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")

    @staticmethod
    def align_crops_from_BB_and_mask_co_tracker(frames,
                                                initial_BB_XYXY,
                                                mask=None,
                                                grid_size=5):
        ### Get Points Grid On BB (For Following): ###
        reference_frame = frames[0]
        if mask is None:
            initial_points = Tracker.generate_points_in_BB(initial_BB_XYXY)  # Generate predicted points
        else:
            bounding_box, polygon_points, initial_points = Tracker.generate_points_in_segmentation_mask(mask, grid_size)

        ### Draw Points On Image (DEBUGGIN): ###
        image_with_points = Tracker.draw_circles_on_image(initial_points, reference_frame, color=(0, 255, 0), radius=5, thickness=2)
        # display_media(image_with_points)

        ### Track points using co_tracker!!!!!: ###
        initial_points_list = initial_points.tolist()
        frames_array = np.concatenate([numpy_unsqueeze(frames[i], 0) for i in range(len(frames))], axis=0)
        pred_tracks, pred_visibility, frames_array = track_points_in_video_auto(frames_array,
                                                                                points=initial_points_list,
                                                                                grid_size=grid_size,
                                                                                interactive=False,
                                                                                fill_all=False)
        predicted_points_list = [pred_tracks[i] for i in range(len(pred_tracks))]
        # imshow_video(frames_output, FPS=5)

        ### Show Points As Tracked With Co-Tracker (DEBUGGING): ###
        frames_with_points_list = []
        for i in np.arange(len(predicted_points_list)):
            current_frame_points = predicted_points_list[i]
            current_frame = frames[i]
            image_with_points = Tracker.draw_circles_on_image(current_frame_points, current_frame, color=(0, 255, 0), radius=5, thickness=2)
            frames_with_points_list.append(image_with_points)
        # frames_with_points_numpy = list_to_numpy(frames_with_points_list)
        # plt.figure(); plt.imshow(frames_with_points_list[-1]);
        # imshow_video(frames_with_points_numpy, FPS=1, frame_stride=1, video_title='tracked points with co-tracker')
        # display_media(frames_with_points_list[-1])

        ### Get Homography From Current Points And Reference Points: ###
        H_list = Tracker.find_homographies_from_points_list(predicted_points_list,
                                                               initial_points,
                                                               method='ransac',  # 'ransac', 'least_squares', 'weighted_least_squares', 'iterative_reweighted_least_squares'
                                                               max_iterations=2000,
                                                               inlier_threshold=3)

        ### Get Points After Applying Homography: ###
        # predicted_points_list_after_homography = Tracker.apply_homographies_to_list_of_points_arrays(predicted_points_list, H_list)
        predicted_points_list_after_homography = []
        bounding_box_list = []
        H,W = frames[0].shape[0:2]
        max_BB_size = -1
        max_BB_shape = (0,0)
        BB_size_factor_list = []
        for i in np.arange(len(frames)):
            bounding_box, segmentation_mask = Tracker.points_to_bounding_box_and_mask(predicted_points_list[i], (H,W))
            current_points = Tracker.generate_points_in_BB(bounding_box, grid_size=5)
            predicted_points_list_after_homography.append(current_points)
            bounding_box_list.append(bounding_box)
            X0,Y0,X1,Y1 = bounding_box
            BB_W = X1 - X0
            BB_H = Y1 - Y0
            BB_size = BB_W * BB_H
            if i == 0:
                BB_original_size = BB_size
            if BB_size > max_BB_size:
                max_BB_size = BB_size
                max_BB_shape = (BB_W, BB_H)
            BB_size_factor = np.sqrt(BB_size / BB_original_size)
            BB_size_factor_list.append(BB_size_factor)

        ### Get Crops From Bounding Boxes (instead of homography): ###
        aligned_crops = []
        for i in np.arange(len(frames)):
            current_BB = bounding_box_list[i]
            original_BB = initial_BB_XYXY
            X0,Y0,X1,Y1 = current_BB
            original_X0,original_Y0,original_X1,original_Y1 = original_BB
            BB_W = X1 - X0
            BB_H = Y1 - Y0
            original_BB_W = original_X1 - original_X0
            original_BB_H = original_Y1 - original_Y0
            Xc = (X0 + X1) / 2
            Yc = (Y0 + Y1) / 2
            BB_W = int(original_BB_W * BB_size_factor_list[i])
            BB_H = int(original_BB_H * BB_size_factor_list[i])
            BB_XYXY = BB_convert_notation_XcYcWH_to_XYXY((Xc,Yc,BB_W,BB_H), (H,W))
            BB_XYXY = [int(BB_XYXY[0]), int(BB_XYXY[1]), int(BB_XYXY[2]), int(BB_XYXY[3])]
            X0,Y0,X1,Y1 = BB_XYXY
            current_frame = frames[i]
            current_crop = current_frame[Y0:Y1, X0:X1]
            max_BB_size_factor = max(BB_size_factor_list)
            final_shape = (int(original_BB_W*max_BB_size_factor), int(original_BB_H*max_BB_size_factor))
            current_crop = cv2.resize(current_crop, final_shape)
            aligned_crops.append(current_crop)
        # display_media(aligned_crops[0])

        ### Show Points As Tracked With Co-Tracker After Homography (DEBUGGING): ###
        frames_with_points_list = []
        for i in np.arange(len(predicted_points_list_after_homography)):
            current_frame_points = predicted_points_list_after_homography[i]
            current_frame = frames[i]
            image_with_points = Tracker.draw_circles_on_image(current_frame_points, current_frame, color=(0, 255, 0), radius=2, thickness=2)
            frames_with_points_list.append(image_with_points)
        # frames_with_points_numpy = list_to_numpy(frames_with_points_list)
        # plt.figure(); plt.imshow(frames_with_points_list[14]);
        # imshow_video(frames_with_points_numpy, FPS=1, frame_stride=1, video_title='tracked points with co-tracker')
        # display_media(frames_with_points_list[5])

        # ### Get and Align Crops Using Homographies: ###
        # input_dict = None
        # aligned_crops = Tracker.align_frame_crops_using_given_homographies(input_dict, frames, frames[0], initial_BB_XYXY, H_list, size_factor=1)
        # imshow_video(list_to_numpy(aligned_crops), FPS=2, frame_stride=1)
        # imshow_np(aligned_crops[-2])
        # imshow_video(list_to_numpy(frames), FPS=5, frame_stride=1)
        # imshow_np(averaged_crop, title='Averaged Crop')
        # display_media(aligned_crops[0])

        return aligned_crops

    @staticmethod
    def draw_bounding_box(image):
        """
        Allow user to draw a bounding box on the given image.

        Args:
            image (np.ndarray): The image to draw the bounding box on.

        Returns:
            tuple: The bounding box coordinates (x0, y0, x1, y1).
        """

        ### Allow User to Draw Bounding Box: ###
        bbox = cv2.selectROI("Select Bounding Box", image, fromCenter=False, showCrosshair=True)  # Draw bounding box
        cv2.destroyAllWindows()  # Close the window
        return BB_convert_notation_XYWH_to_XYXY(bbox)  # Return bounding box coordinates

    @staticmethod
    def draw_circles_on_image(points_array, input_image, color=(0, 255, 0), radius=5, thickness=2):
        """
        Draw circles on the input image at the specified points.

        Args:
            points_array (np.ndarray): Array of points with shape [N, 2].
            input_image (np.ndarray): The input image on which to draw the circles.
            color (tuple): Color of the circles (default is green).
            radius (int): Radius of the circles (default is 5).
            thickness (int): Thickness of the circle outlines (default is 2).

        Returns:
            np.ndarray: Image with circles drawn on it.
        """
        output_image = input_image.copy()  # Create a copy of the input image to draw on

        ### Drawing Circles: ###
        for point in points_array:  # Loop through each point
            x, y = int(point[0]), int(point[1])  # Extract coordinates and convert to integers
            cv2.circle(output_image, (x, y), radius, color, thickness)  # Draw circle at the specified point

        return output_image  # Return the image with circles drawn on it

    @staticmethod
    def align_frame_crops_using_given_homographies(input_dict: dict,
                                                   frames: list = None,
                                                   reference_frame: np.ndarray = None,
                                                   bounding_box: list = None,
                                                   homographies: list = None,
                                                   size_factor: float = 1.0) -> dict:
        """
        Align crops using a bounding box and homography matrices, then compute the averaged crop.

        Args:
            input_dict (dict): Dictionary containing the following keys:
                - 'frames' (list or tensor): List of frames (numpy arrays or PyTorch tensors). Each frame shape is [H, W, C].
                - 'reference_frame' (np.ndarray or tensor): Reference frame with shape [H, W, C].
                - 'bounding_box' (list): Bounding box coordinates on the reference frame [X0, Y0, X1, Y1].
                - 'homographies' (list): List of homography matrices for each frame. Each homography is a 3x3 numpy array.
                - 'size_factor' (float): Factor by which to expand the bounding box size. Default is 1.0.
            frames (list, optional): List of frames to align. Takes precedence over input_dict if provided.
            reference_frame (np.ndarray, optional): Reference frame. Takes precedence over input_dict if provided.
            bounding_box (list, optional): Bounding box coordinates. Takes precedence over input_dict if provided.
            homographies (list, optional): List of homography matrices. Takes precedence over input_dict if provided.
            size_factor (float, optional): Size factor to expand bounding box. Takes precedence over input_dict if provided.

        Returns:
            dict: Dictionary containing the following keys:
                - 'aligned_crops' (list): List of aligned crops.
                - 'averaged_crop' (np.ndarray): Averaged aligned crop.
        """
        ### Extract Inputs: ###
        frames = frames if frames is not None else input_dict.get('frames')  # Extract list of frames
        reference_frame = reference_frame if reference_frame is not None else input_dict.get('reference_frame')  # Extract reference frame
        bounding_box = bounding_box if bounding_box is not None else input_dict.get('bounding_box')  # Extract bounding box
        homographies = homographies if homographies is not None else input_dict.get('homographies')  # Extract list of homography matrices
        size_factor = size_factor if size_factor is not None else input_dict.get('size_factor', 1.0)  # Extract size factor, default to 1.0 if not provided

        ### Convert frames and reference_frame to numpy arrays if they are tensors ###
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.cpu().numpy() for frame in frames]  # Convert each frame to numpy array
        if isinstance(reference_frame, torch.Tensor):
            reference_frame = reference_frame.cpu().numpy()  # Convert reference frame to numpy array

        ### Extract bounding_box coordinates ###
        x0, y0, x1, y1 = bounding_box  # Extract bounding box coordinates

        ### Calculate the expanded bounding box coordinates ###
        box_width = x1 - x0  # Calculate box width
        box_height = y1 - y0  # Calculate box height
        center_x = x0 + box_width // 2  # Calculate center x-coordinate
        center_y = y0 + box_height // 2  # Calculate center y-coordinate
        new_width = int(box_width * size_factor)  # Calculate new width
        new_height = int(box_height * size_factor)  # Calculate new height
        new_x0 = max(center_x - new_width // 2, 0)  # Calculate new top-left x-coordinate
        new_y0 = max(center_y - new_height // 2, 0)  # Calculate new top-left y-coordinate
        new_x1 = min(center_x + new_width // 2, reference_frame.shape[1])  # Calculate new bottom-right x-coordinate
        new_y1 = min(center_y + new_height // 2, reference_frame.shape[0])  # Calculate new bottom-right y-coordinate

        ### Initialize list to store aligned crops ###
        aligned_crops = []

        ### Looping Over Indices: ###
        for i, H in enumerate(homographies):  # Loop through each frame and its corresponding homography matrix
            h, w = reference_frame.shape[:2]  # Get height and width from reference frame shape
            aligned_frame = cv2.warpPerspective(frames[i], H, (w, h))  # Apply homography to align frame
            crop = aligned_frame[new_y0:new_y1, new_x0:new_x1]  # Crop the aligned frame using expanded bounding box coordinates
            aligned_crops.append(crop)  # Add crop to list

        ### Compute the averaged crop ###
        averaged_crop = np.mean(aligned_crops, axis=0).astype(np.uint8)  # Compute the averaged crop

        ### Prepare Output Dictionary: ###
        output_dict = {
            'aligned_crops': aligned_crops,  # List of aligned crops
            'averaged_crop': averaged_crop  # Averaged crop
        }
        return aligned_crops  # Return output dictionary

    @staticmethod
    def apply_homography_to_points_array(points, homography):
        """
        Applies a homography matrix to a set of points.

        Inputs:
        - points: numpy array of shape [N, 2], where N is the number of points
        - homography: numpy array of shape [3, 3], the homography matrix

        Outputs:
        - transformed_points: numpy array of shape [N, 2], the transformed points
        """

        # Convert points to homogeneous coordinates
        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])  # Shape [N, 3]

        # Apply homography matrix
        transformed_points_homogeneous = np.dot(homography, homogeneous_points.T).T  # Shape [N, 3]

        # Convert back to Cartesian coordinates
        transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2].reshape(-1,
                                                                                                                  1)  # Shape [N, 2]

        return transformed_points

    @staticmethod
    def apply_homographies_to_list_of_points_arrays(points_list, homographies_list):
        """
        Applies a list of homography matrices to a list of point arrays.

        Inputs:
        - points_list: list of numpy arrays, each of shape [N, 2], where N is the number of points
        - homographies: list of numpy arrays, each of shape [3, 3], the homography matrices

        Outputs:
        - transformed_points_list: list of numpy arrays, each of shape [N, 2], the transformed points
        """

        # Check if the length of points_list and homographies are the same
        if len(points_list) != len(homographies_list):
            raise ValueError("The length of points_list and homographies must be the same.")

        # Apply homography to each set of points
        transformed_points_list = [Tracker.apply_homography_to_points_array(points, homography) for points, homography in
                                   zip(points_list, homographies_list)]

        return transformed_points_list

    @staticmethod
    def find_homographies_from_points_list(predicted_points_array, reference_points, method='ransac', max_iterations=100, inlier_threshold=3):
        """
        Calculate homographies from a list of predicted points using the specified method.

        Formerly named: calculate_homographies

        Inputs:
        - reference_points: numpy array of reference points of shape [N, 2]
        - predicted_points_array: numpy array of predicted points for each frame of shape [T, N, 2] or list of [N, 2]
        - method: str, method to estimate homography ('ransac' or 'wls')

        Outputs:
        - homographies: list of homography matrices
        """

        ### Initialize List for Homographies: ###
        homographies = []  # Initialize list for homographies

        ### Looping Over Indices: ###
        for i in range(len(predicted_points_array)):  # Iterate through predicted points
            if method == 'ransac':  # If method is RANSAC
                H, _ = cv2.findHomography(predicted_points_array[i], reference_points, cv2.RANSAC, maxIters=max_iterations)  # Estimate homography using RANSAC
            elif method == 'least_squares':  # If method is
                H = Tracker.find_homography_for_two_point_sets_simple_least_squares(predicted_points_array[i], reference_points)  # Estimate homography using LS
            elif method == 'weighted_least_squares':  # If method is WLS
                H = Tracker.find_homography_for_two_point_sets_weighted_least_squares(predicted_points_array[i], reference_points)
            elif method == 'iterative_reweighted_least_squares':  # If method is WLS
                H = Tracker.find_homography_for_two_point_sets_iterative_reweighted_least_squares(predicted_points_array[i],
                                                                                                     reference_points,
                                                                                                     max_iterations=max_iterations,
                                                                                                     inlier_threshold=inlier_threshold)
            homographies.append(H)  # Append homography

        return homographies  # Return list of homographies

    @staticmethod
    def find_homography_for_two_point_sets_simple_least_squares(src_points, dst_points):
        """
        Find homography using simple least squares.

        Formerly named: find_homography_wls

        Inputs:
        - src_points: numpy array of source points of shape [N, 2]
        - dst_points: numpy array of destination points of shape [N, 2]

        Outputs:
        - H: homography matrix of shape [3, 3]
        """

        ### Initialize Matrices A and B: ###
        A = []  # Initialize matrix A
        B = []  # Initialize matrix B

        ### Looping Over Indices: ###
        for i in range(len(src_points)):  # Iterate through source points
            x, y = src_points[i]  # Source point coordinates
            u, v = dst_points[i]  # Destination point coordinates
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])  # Append to A
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y])  # Append to A
            B.append(u)  # Append u to B
            B.append(v)  # Append v to B

        ### Convert A and B to Numpy Arrays: ###
        A = np.array(A)  # Convert A to numpy array
        B = np.array(B)  # Convert B to numpy array

        ### Solve for H: ###
        H = np.linalg.lstsq(A, B, rcond=None)[0]  # Solve for H
        H = np.append(H, 1).reshape(3, 3)  # Reshape H to 3x3

        return H  # Return homography matrix

    @staticmethod
    def find_homography_for_two_point_sets_weighted_least_squares(src_pts, dst_pts, weights):
        """
        Calculate homography matrix using weighted least squares.

        Parameters:
        - src_pts (np.ndarray): Source points.
        - dst_pts (np.ndarray): Destination points.
        - weights (np.ndarray): Weights for each point.

        Returns:
        - H (np.ndarray): Homography matrix.
        """

        A = []  ### Initialize list to store A matrix
        B = []  ### Initialize list to store B vector

        ### Construct A matrix and B vector: ###
        for (x1, y1), (x2, y2), w in zip(src_pts, dst_pts, weights):
            A.append([x1, y1, 1, 0, 0, 0, -w * x1 * x2, -w * y1 * x2])  ### Append row to A matrix
            A.append([0, 0, 0, x1, y1, 1, -w * x1 * y2, -w * y1 * y2])  ### Append row to A matrix
            B.append(x2)  ### Append x2 to B vector
            B.append(y2)  ### Append y2 to B vector

        A = np.array(A)  ### Convert A list to numpy array
        B = np.array(B)  ### Convert B list to numpy array

        ### Solve for h in Ah = B: ###
        h, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  ### Solve least squares problem
        H = np.append(h, 1).reshape(3, 3)  ### Reshape h to 3x3 homography matrix
        return H  ### Return homography matrix

    @staticmethod
    def find_homography_for_two_point_sets_iterative_reweighted_least_squares(src_pts, dst_pts, max_iterations=2000, inlier_threshold=5.0, c=4.685):
        """
        Aligns and refines the homography matrix using RANSAC and weighted least squares.

        Inputs:
        - src_pts: numpy array of source points of shape [N, 2]
        - dst_pts: numpy array of destination points of shape [N, 2]
        - max_iterations: int, maximum number of RANSAC iterations
        - inlier_threshold: float, threshold to determine inliers
        - c: float, tuning constant for Tukey's Biweight function

        Outputs:
        - H: numpy array, refined homography matrix of shape [3, 3]
        """

        ### Initialize Variables: ###
        for iteration in range(max_iterations):  # Iterate through maximum iterations

            ### Find Homography Using RANSAC: ###
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0,
                                         maxIters=2000)  # Find homography using RANSAC
            if H is None:  # If homography is None
                break  # Break the loop

            ### Get RANSAC Inliers: ###
            RANSAC_inliers = mask.ravel().astype(bool)  # Get RANSAC inliers
            num_inliers = np.sum(RANSAC_inliers)  # Count number of inliers
            if num_inliers < 4:  # If number of inliers is less than 4
                break  # Break the loop

            ### Transform Source Points and Calculate Residuals: ###
            transformed_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)  # Transform source points
            residuals = np.linalg.norm(src_pts - transformed_pts, axis=1)  # Calculate residuals
            outliers = residuals > inlier_threshold  # Identify outliers
            RANSAC_inliers[RANSAC_inliers] = ~outliers[RANSAC_inliers]  # Update RANSAC inliers

            ### Reweight Inliers Using Tukey's Biweight Function: ###
            weights = tukey_biweight(residuals, c)  # Reweight inliers using Tukey's Biweight function
            weights = weights / np.sum(weights)  # Normalize weights

            ### Get Inlier Points and Weights: ###
            src_pts_inliers = src_pts[RANSAC_inliers]  # Get inlier source points
            dst_pts_inliers = dst_pts[RANSAC_inliers]  # Get inlier destination points
            weights_inliers = weights[RANSAC_inliers]  # Get inlier weights

            ### Update Homography Using Weighted Least Squares: ###
            H = Tracker.weighted_least_squares_homography_matrix_points(src_pts_inliers, dst_pts_inliers, weights_inliers)  # Update homography using weighted least squares

            ### Update Points and Weights for Next Iteration: ###
            src_pts = src_pts[RANSAC_inliers]  # Update source points for next iteration
            dst_pts = dst_pts[RANSAC_inliers]  # Update destination points for next iteration
            weights = weights[RANSAC_inliers]  # Update weights for next iteration

            ### Check for Convergence: ###
            if num_inliers == np.sum(RANSAC_inliers):  # If number of inliers does not change
                break  # Break the loop

        return H  # Return refined homography matrix

    @staticmethod
    def weighted_least_squares_homography_matrix_points(src_pts, dst_pts, weights):
        """
        Calculate homography matrix using weighted least squares.

        Parameters:
        - src_pts (np.ndarray): Source points.
        - dst_pts (np.ndarray): Destination points.
        - weights (np.ndarray): Weights for each point.

        Returns:
        - H (np.ndarray): Homography matrix.
        """

        A = []  ### Initialize list to store A matrix
        B = []  ### Initialize list to store B vector

        ### Construct A matrix and B vector: ###
        for (x1, y1), (x2, y2), w in zip(src_pts, dst_pts, weights):
            A.append([x1, y1, 1, 0, 0, 0, -w * x1 * x2, -w * y1 * x2])  ### Append row to A matrix
            A.append([0, 0, 0, x1, y1, 1, -w * x1 * y2, -w * y1 * y2])  ### Append row to A matrix
            B.append(x2)  ### Append x2 to B vector
            B.append(y2)  ### Append y2 to B vector

        A = np.array(A)  ### Convert A list to numpy array
        B = np.array(B)  ### Convert B list to numpy array

        ### Solve for h in Ah = B: ###
        h, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  ### Solve least squares problem
        H = np.append(h, 1).reshape(3, 3)  ### Reshape h to 3x3 homography matrix
        return H  ### Return homography matrix


    def get_center_point_from_BB_or_mask(self, mask, bbox):
        """
        Generate grid points within the mask or at the center of the bounding box if the mask is empty.

        Args:
            mask (np.ndarray): Binary mask of shape [H, W].
            bbox (tuple): Bounding box coordinates (x, y, w, h).

        Returns:
            list: List of grid points, each represented as a tuple (center_x, center_y).

        Attributes:
            self: Instance of the class containing this method.

        This function performs the following steps:
            1. Finds the coordinates of the non-zero elements in the mask.
            2. Calculates the center of the mask or the bounding box if the mask is empty.
            3. Returns the center point as a list.
        """

        ### Finding Non-Zero Elements in the Mask: ###
        y, x = np.where(mask)  # Find the coordinates of non-zero elements in the mask

        if len(x) == 0 or len(y) == 0:  # Check if the mask is empty
            ### Use Center of Bounding Box if Mask is Empty: ###
            center_x = bbox[0] + bbox[2] // 2  # Calculate center x-coordinate of the bounding box
            center_y = bbox[1] + bbox[3] // 2  # Calculate center y-coordinate of the bounding box
        else:
            ### Calculate Center of the Mask: ###
            center_x, center_y = np.mean(x), np.mean(y)  # Calculate mean x and y coordinates of the mask

        return [(center_x, center_y)]  # Return the center point as a list

    def process_frames_auto(self, original_bbox, tracks, visibility):
        """
        Process frames using tracked points to generate cropped frames.

        Args:
            original_bbox (tuple): Original bounding box coordinates (x, y, w, h).
            tracks (np.ndarray): Tracked points for each frame.
            visibility (np.ndarray): Visibility of tracked points for each frame.

        Attributes:
            self.trimmed_frames (list): List of trimmed frames, each of shape [H, W, C].
            self.processed_frames (list): List of processed frames, each of shape [H, W, C].
            self.current_frame (int): Index of the current frame being displayed.
            self.frame_slider (QSlider): Slider widget to navigate through frames.
            self.frames (list): List of frames to display.

        This function performs the following steps:
            1. Displays a progress dialog.
            2. Iterates over each frame and tracks.
            3. Crops the frame based on the tracked points and the original bounding box size.
            4. Ensures the cropped frame does not go out of frame boundaries.
            5. Updates the processed frames and frame slider.
        """

        ### This Is The Code Block: ###
        self.show_progress_dialog("Processing frames...")  # Display progress dialog

        try:
            self.processed_frames = []  # Initialize list to store processed frames
            height, width = self.trimmed_frames[0].shape[:2]  # Get height and width of the frame
            orig_x, orig_y, orig_w, orig_h = original_bbox  # Unpack original bounding box

            logging.debug(f"Original bbox (pixels): x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")  # Log original bounding box

            ### Looping Over Indices: ###
            for i, frame in enumerate(self.trimmed_frames):  # Loop through each frame
                center_x, center_y = tracks[i][0]  # Get the center point from the tracks for the current frame

                ### Calculate New Bounding Box Coordinates: ###
                new_x = int(center_x - orig_w / 2)  # Calculate new x-coordinate
                new_y = int(center_y - orig_h / 2)  # Calculate new y-coordinate

                ### Ensure the Box Doesn't Go Out of Frame: ###
                new_x = max(0, min(new_x, width - orig_w))  # Ensure new_x is within frame width
                new_y = max(0, min(new_y, height - orig_h))  # Ensure new_y is within frame height

                logging.debug(f"Frame {i}: Center ({center_x}, {center_y}), New top-left: ({new_x}, {new_y})")  # Log new coordinates

                ### Crop the Frame: ###
                cropped_frame = frame[new_y:new_y + orig_h, new_x:new_x + orig_w]  # Crop the frame based on new coordinates
                self.processed_frames.append(cropped_frame)  # Add cropped frame to list

            ### Update Frame Slider: ###
            self.current_frame = 0  # Set current frame to 0
            self.frame_slider.setRange(0, len(self.processed_frames) - 1)  # Set range for frame slider
            self.frame_slider.setValue(0)  # Set initial value for frame slider
            self.frames = self.processed_frames  # Update frames to display
            self.update_frame()  # Update the displayed frame
            logging.debug(f"Processed {len(self.processed_frames)} frames")  # Log the number of processed frames

        finally:
            self.close_progress_dialog()  # Close progress dialog

    @staticmethod
    def polygon_to_bounding_box_and_mask(polygon_points, input_shape):
        """
        This function takes a list of polygon points and returns:
        1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
        2. A binary segmentation mask of the input shape with 1s inside the polygon.

        Former function name: N/A

        Parameters:
        polygon_points (list): A list of tuples representing the polygon points.
                               Each tuple contains two integers (x, y).
        input_shape (tuple): A tuple representing the shape of the input (height, width).

        Returns:
        bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                              X0, Y0 are the coordinates of the top-left corner.
                              X1, Y1 are the coordinates of the bottom-right corner.
        segmentation_mask (np.ndarray): A binary mask of the same shape as the input, with the polygon area filled with ones.
                                        Shape is (height, width).
        """

        ### Calculate Bounding Box: ###
        x_coords, y_coords = zip(*polygon_points)  # unzip the polygon points into x and y coordinates
        X0, Y0 = min(x_coords), min(y_coords)  # calculate the minimum x and y coordinates
        X1, Y1 = max(x_coords), max(y_coords)  # calculate the maximum x and y coordinates
        bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

        ### Initialize Segmentation Mask: ###
        segmentation_mask = np.zeros(input_shape,
                                     dtype=np.uint8)  # create an empty mask with the same shape as the input

        ### Insert 1s in Polygon Area: ###
        rr, cc = polygon(y_coords, x_coords)  # get the row and column indices of the polygon
        segmentation_mask[rr, cc] = 1  # fill the polygon area with 1s

        return bounding_box, segmentation_mask  # return the bounding box and segmentation mask

    @staticmethod
    def bounding_box_to_polygon_and_mask(bounding_box, input_shape):
        """
        This function takes a bounding box in the format (X0, Y0, X1, Y1) and returns:
        1. A list of polygon points from the bounding box edges.
        2. A binary segmentation mask of the input shape with 1s inside the bounding box.

        Former function name: N/A

        Parameters:
        bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                              X0, Y0 are the coordinates of the top-left corner.
                              X1, Y1 are the coordinates of the bottom-right corner.
        input_shape (tuple): A tuple representing the shape of the input (height, width).

        Returns:
        polygon_points (list): A list of tuples representing the four corners of the bounding box.
                               Each tuple contains two integers (x, y).
        segmentation_mask (np.ndarray): A binary mask of the same shape as the input, with the bounding box area filled with ones.
                                        Shape is (height, width).
        """

        ### Extract Bounding Box Coordinates: ###
        X0, Y0, X1, Y1 = bounding_box  # unpack bounding box coordinates

        ### Create List of Polygon Points: ###
        polygon_points = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]  # create list of the four corners of the bounding box

        ### Initialize Segmentation Mask: ###
        segmentation_mask = np.zeros(input_shape,
                                     dtype=np.uint8)  # create an empty mask with the same shape as the input

        ### Insert 1s in Bounding Box Area: ###
        segmentation_mask[Y0:Y1, X0:X1] = 1  # fill the bounding box area with 1s using slicing

        return polygon_points, segmentation_mask  # return the polygon points and segmentation mask

    @staticmethod
    def mask_to_bounding_box_and_polygon(segmentation_mask):
        """
        This function takes a segmentation mask and returns:
        1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
        2. A list of polygon points representing the contour of the mask area.

        Former function name: N/A

        Parameters:
        segmentation_mask (np.ndarray): A binary mask with the shape (height, width), where 1s represent the mask area.

        Returns:
        bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                              X0, Y0 are the coordinates of the top-left corner.
                              X1, Y1 are the coordinates of the bottom-right corner.
        polygon_points (list): A list of tuples representing the contour points of the mask area.
                               Each tuple contains two integers (x, y).
        """

        ### Find Contours in the Segmentation Mask: ###
        contours = find_contours(segmentation_mask, level=0.5)  # find contours at the mask boundaries

        ### Check if Contours Were Found: ###
        if len(contours) == 0:
            raise ValueError("No contours found in the segmentation mask.")

        ### Extract the Largest Contour: ###
        largest_contour = max(contours, key=len)  # select the largest contour by length

        ### Convert Contour to Integer Coordinates: ###
        polygon_points = [(int(x), int(y)) for y, x in largest_contour]  # convert contour to integer coordinates

        ### Calculate Bounding Box: ###
        x_coords, y_coords = zip(*polygon_points)  # unzip the polygon points into x and y coordinates
        X0, Y0 = min(x_coords), min(y_coords)  # calculate the minimum x and y coordinates
        X1, Y1 = max(x_coords), max(y_coords)  # calculate the maximum x and y coordinates
        bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

        return bounding_box, polygon_points  # return the bounding box and polygon points

    @staticmethod
    def points_to_bounding_box_and_mask(points, mask_shape):
        """
        This function takes a grid of points and returns:
        1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
        2. A binary segmentation mask with 1s inside the bounding box.

        Parameters:
        points (np.ndarray): A 2D array of shape [M, 2] where M is the number of points and each point is (x, y).
        mask_shape (tuple): A tuple representing the shape of the output mask (height, width).

        Returns:
        bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                              X0, Y0 are the coordinates of the top-left corner.
                              X1, Y1 are the coordinates of the bottom-right corner.
        segmentation_mask (np.ndarray): A binary mask of shape (height, width) with 1s inside the bounding box.
        """

        ### Extract X and Y Coordinates: ###
        x_coords, y_coords = points[:, 0], points[:, 1]  # extract x and y coordinates from the points

        ### Calculate Bounding Box: ###
        X0, Y0 = int(np.min(x_coords)), int(np.min(y_coords))  # calculate the minimum x and y coordinates
        X1, Y1 = int(np.max(x_coords)), int(np.max(y_coords))  # calculate the maximum x and y coordinates
        bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

        ### Create Segmentation Mask: ###
        segmentation_mask = np.zeros(mask_shape, dtype=np.uint8)  # initialize a mask with zeros
        segmentation_mask[Y0:Y1 + 1, X0:X1 + 1] = 1  # set the area inside the bounding box to 1s

        return bounding_box, segmentation_mask  # return the bounding box and segmentation mask

    @staticmethod
    def generate_points_in_BB(bbox, grid_size=5):
        """
        Generate a grid of points within the bounding box.

        Args:
            bbox (tuple or np.ndarray): The bounding box coordinates (x0, y0, x1, y1).
            grid_size (int): The number of points along each dimension.

        Returns:
            list or np.ndarray: The grid of points with shape [grid_size*grid_size, 2]. If bbox is an array of shape [N, 4],
                                returns a list of grids for each bounding box.
        """

        def generate_grid_points(x0, y0, x1, y1, grid_size):
            """
            Helper function to generate grid points within a single bounding box.
            """
            x = np.linspace(x0, x1, grid_size)  # Generate x-coordinates
            y = np.linspace(y0, y1, grid_size)  # Generate y-coordinates
            points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)  # Generate grid of points
            return points  # Return grid of points

        ### Handling Different BBox Formats: ###
        if isinstance(bbox, tuple) or isinstance(bbox, list):  # If bbox is a tuple
            x0, y0, x1, y1 = bbox  # Extract bounding box coordinates
            points = generate_grid_points(x0, y0, x1, y1, grid_size)  # Generate grid points
            return points  # Return grid of points

        elif isinstance(bbox, np.ndarray) and bbox.shape == (4,):  # If bbox is a 1D array of size 4
            x0, y0, x1, y1 = bbox  # Extract bounding box coordinates
            points = generate_grid_points(x0, y0, x1, y1, grid_size)  # Generate grid points
            return points  # Return grid of points

        elif isinstance(bbox, np.ndarray) and bbox.shape[1] == 4:  # If bbox is a 2D array of shape [N, 4]
            points_list = []
            ### Looping Over Bounding Boxes: ###
            for single_bbox in bbox:  # Loop through each bounding box
                x0, y0, x1, y1 = single_bbox  # Extract bounding box coordinates
                points = generate_grid_points(x0, y0, x1, y1, grid_size)  # Generate grid points
                points_list.append(points)  # Append points to list
            return points_list  # Return list of grids for each bounding box

        else:
            raise ValueError("Invalid bbox format. Must be a tuple or an array of shape [N, 4] or [4,].")

    @staticmethod
    def generate_points_in_polygon(polygon, grid_size=5):
        """
        Generate a grid of points within a closed polygon.

        Args:
            polygon (list or np.ndarray): The polygon vertices as a list of tuples or an array of shape [N, 2].
            grid_size (int): The number of points along the longer dimension of the bounding box.

        Returns:
            np.ndarray: The grid of points within the polygon with shape [M, 2], where M is the number of points inside the polygon.
        """
        ### This Is The Code Block: ###
        polygon = np.array(polygon)  # Convert polygon to numpy array if it is not already
        x_min, y_min = np.min(polygon, axis=0)  # Get the minimum x and y coordinates
        x_max, y_max = np.max(polygon, axis=0)  # Get the maximum x and y coordinates

        ### Create a grid of points within the bounding box of the polygon ###
        x_points = np.linspace(x_min, x_max, grid_size)  # Generate x-coordinates
        y_points = np.linspace(y_min, y_max, grid_size)  # Generate y-coordinates
        x_grid, y_grid = np.meshgrid(x_points, y_points)  # Create meshgrid of x and y points
        grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T  # Flatten the grid to a list of points

        ### Check which points are inside the polygon ###
        path = mpath.Path(polygon)  # Create a path object from the polygon
        inside_mask = path.contains_points(grid_points)  # Check which grid points are inside the polygon
        points_in_polygon = grid_points[inside_mask]  # Select only the points inside the polygon

        return points_in_polygon  # Return the grid of points within the polygon

    @staticmethod
    def generate_points_in_segmentation_mask(segmentation_mask, grid_size=5):
        """
        This function takes a segmentation mask with one area of 1s and returns:
        1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
        2. A list of polygon points representing the contour of the mask area.
        3. A grid of points within the polygon area inside the mask.

        Former function name: N/A

        Parameters:
        segmentation_mask (np.ndarray): A binary mask with the shape (height, width), where 1s represent the mask area.
        grid_size (int): The number of points along the longer dimension of the bounding box.

        Returns:
        bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                              X0, Y0 are the coordinates of the top-left corner.
                              X1, Y1 are the coordinates of the bottom-right corner.
        polygon_points (list): A list of tuples representing the contour points of the mask area.
                               Each tuple contains two integers (x, y).
        points_in_polygon (np.ndarray): A grid of points within the polygon area inside the mask.
                                        Shape is [M, 2], where M is the number of points inside the polygon.
        """

        ### Find Contours in the Segmentation Mask: ###
        contours = find_contours(segmentation_mask, level=0.5)  # find contours at the mask boundaries

        ### Check if Contours Were Found: ###
        if len(contours) == 0:
            ### Return Default Output: ###
            height, width = segmentation_mask.shape  # extract height and width of the input mask

            ### Define Bounding Box as Entire Frame: ###
            bounding_box = (0, 0, width, height)  # set bounding box to cover the entire frame

            ### Create Segmentation Mask of All 1s: ###
            segmentation_mask = np.ones((height, width), dtype=np.uint8)  # create a mask filled with 1s

            ### Define Polygon as Entire Image Vertices: ###
            polygon_points = [(0, 0), (width - 1, 0), (width - 1, height - 1),
                              (0, height - 1)]  # vertices of the entire image

            ### Generate Points within the Bounding Box: ###
            points_in_polygon = Tracker.generate_points_in_BB(bounding_box, grid_size)  # generate points within the bounding box

            ### Return the Default Values: ###
            return bounding_box, polygon_points, points_in_polygon  # return the default output

        ### Extract the Largest Contour: ###
        largest_contour = max(contours, key=len)  # select the largest contour by length

        ### Convert Contour to Integer Coordinates: ###
        polygon_points = [(int(x), int(y)) for y, x in largest_contour]  # convert contour to integer coordinates

        ### Calculate Bounding Box: ###
        x_coords, y_coords = zip(*polygon_points)  # unzip the polygon points into x and y coordinates
        X0, Y0 = min(x_coords), min(y_coords)  # calculate the minimum x and y coordinates
        X1, Y1 = max(x_coords), max(y_coords)  # calculate the maximum x and y coordinates
        bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

        ### Generate Points within the Polygon: ###
        points_in_polygon = Tracker.generate_points_in_polygon(polygon_points, grid_size)  # generate points within the polygon

        return bounding_box, polygon_points, points_in_polygon  # return the bounding box, polygon points, and points in polygon


def RGB2BW(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def BW2RGB(frame):
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def frames_to_constant_format(frames, dtype_requested='uint8', range_requested=[0, 255], channels_requested=3,
                              threshold=5):
    first_frame = frames[0]
    original_dtype = first_frame.dtype
    original_channels = first_frame.shape[2] if len(first_frame.shape) == 3 else 1

    if original_dtype == np.uint8:
        original_range = [0, 255]
    else:
        max_val = np.max(first_frame)
        original_range = [0, 255] if max_val > threshold else [0, 1]

    processed_frames = []

    for frame in frames:
        if original_channels != channels_requested:
            if channels_requested == 1:
                frame = RGB2BW(frame)
            else:
                frame = BW2RGB(frame)

        if original_range != range_requested:
            if original_range == [0, 255] and range_requested == [0, 1]:
                frame = frame / 255.0
            elif original_range == [0, 1] and range_requested == [0, 255]:
                frame = frame * 255.0

        if original_dtype != dtype_requested:
            frame = frame.astype(dtype_requested)

        processed_frames.append(frame)

    return processed_frames


def video_to_frames(video_path, dtype_requested='uint8', range_requested=[0, 255], channels_requested=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    processed_frames = frames_to_constant_format(frames, dtype_requested, range_requested, channels_requested)
    return processed_frames

if "__main__" == __name__:
    video_path = "C:/Users/dudyk/Downloads/test_for_tracking.mp4"
    frames = video_to_frames(video_path)
    bbox = (1467, 293, 205, 192)

    croped_frames = Tracker.align_crops_from_BB(frames,
                            bbox,
                            tracking_method='co_tracker')


