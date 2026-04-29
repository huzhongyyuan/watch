import torch
import cv2
import numpy as np
from core.utils.wis3d_utils import get_colors_by_conf


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return np.array(x)
    return x.clone().cpu().numpy()


def draw_bbx_xys_on_image(bbx_xys, image, conf=True):
    assert isinstance(bbx_xys, np.ndarray)
    assert isinstance(image, np.ndarray)
    image = image.copy()
    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, lu_point, rd_point, color, 2)
    return image


def draw_bbx_xys_on_image_batch(bbx_xys_batch, image_batch, conf=None):
    """conf: if provided, list of bool"""
    use_conf = conf is not None
    bbx_xys_batch = to_numpy(bbx_xys_batch)
    assert len(bbx_xys_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xys_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i], conf[i]))
        else:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i]))
    return image_batch_out


def draw_bbx_xyxy_on_image(bbx_xys, image, conf=True):
    bbx_xys = to_numpy(bbx_xys)
    image = to_numpy(image)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, (int(bbx_xys[0]), int(bbx_xys[1])), (int(bbx_xys[2]), int(bbx_xys[3])), color, 2)
    return image


def draw_bbx_xyxy_on_image_batch(bbx_xyxy_batch, image_batch, mask=None, conf=None):
    """
    Args:
        conf: if provided, list of bool, mutually exclusive with mask
        mask: whether to draw, historically used
    """
    if mask is not None:
        assert conf is None
    if conf is not None:
        assert mask is None
    use_conf = conf is not None
    bbx_xyxy_batch = to_numpy(bbx_xyxy_batch)
    image_batch = to_numpy(image_batch)
    assert len(bbx_xyxy_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xyxy_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i], conf[i]))
        else:
            if mask is None or mask[i]:
                image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i]))
            else:
                image_batch_out.append(image_batch[i])
    return image_batch_out


def draw_kpts(frame, keypoints, color=(0, 255, 0), thickness=2):
    frame_ = frame.copy()
    for x, y in keypoints:
        cv2.circle(frame_, (int(x), int(y)), thickness, color, -1)
    return frame_


def draw_kpts_with_conf(frame, kp2d, conf, thickness=2):
    """
    Args:
        kp2d: (J, 2),
        conf: (J,)
    """
    frame_ = frame.copy()
    conf = conf.reshape(-1)
    colors = get_colors_by_conf(conf)  # (J, 3)
    colors = colors[:, [2, 1, 0]].int().numpy().tolist()
    for j in range(kp2d.shape[0]):
        x, y = kp2d[j, :2]
        c = colors[j]
        cv2.circle(frame_, (int(x), int(y)), thickness, c, -1)
    return frame_


def draw_kpts_with_conf_batch(frames, kp2d_batch, conf_batch, thickness=2):
    """
    Args:
        kp2d_batch: (B, J, 2),
        conf_batch: (B, J)
    """
    assert len(frames) == len(kp2d_batch)
    assert len(frames) == len(conf_batch)
    frames_ = []
    for i in range(len(frames)):
        frames_.append(draw_kpts_with_conf(frames[i], kp2d_batch[i], conf_batch[i], thickness))
    return frames_


def draw_coco17_skeleton(img, keypoints, conf_thr=0):
    """绘制彩色COCO17骨架，左右部位使用不同颜色（精简版）"""
    use_conf_thr = keypoints.shape[1] == 3
    img = img.copy()
    # fmt:off
    coco_skel = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]            
    # fmt:on

    # 定义左右部位颜色 (BGR格式)
    LEFT_COLOR = (0, 0, 255)    # 红色 - 左半身
    RIGHT_COLOR = (0, 255, 0)   # 绿色 - 右半身
    CENTER_COLOR = (255, 0, 0)  # 蓝色 - 躯干

    # 动态分配颜色函数
    def get_color_for_point(idx):
        if idx in [1, 3, 5, 7, 9, 11, 13, 15]:  # 左半身关键点
            return LEFT_COLOR
        elif idx in [2, 4, 6, 8, 10, 12, 14, 16]:  # 右半身关键点
            return RIGHT_COLOR
        return CENTER_COLOR  # 躯干关键点

    # 绘制骨骼
    for bone in coco_skel:
        # 动态确定骨骼颜色
        color = CENTER_COLOR
        if all(idx in [1, 3, 5, 7, 9, 11, 13, 15] for idx in bone):
            color = LEFT_COLOR
        elif all(idx in [2, 4, 6, 8, 10, 12, 14, 16] for idx in bone):
            color = RIGHT_COLOR

        if use_conf_thr:
            kp1, kp2 = keypoints[bone[0]], keypoints[bone[1]]
            if kp1[2] > conf_thr and kp2[2] > conf_thr:
                img = cv2.line(img,
                              (int(kp1[0]), int(kp1[1])),
                              (int(kp2[0]), int(kp2[1])),
                              color, 4)
        else:
            kp1, kp2 = keypoints[bone[0]], keypoints[bone[1]]
            img = cv2.line(img,
                          (int(kp1[0]), int(kp1[1])),
                          (int(kp2[0]), int(kp2[1])),
                          color, 4)

    # 绘制关键点
    for i, kp in enumerate(keypoints):
        if use_conf_thr and kp[2] <= conf_thr:
            continue

        color = get_color_for_point(i)
        img = cv2.circle(img, (int(kp[0]), int(kp[1])), 6, color, -1)

    return img


def draw_coco17_skeleton_batch(imgs, keypoints_batch, conf_thr=0):
    assert len(imgs) == len(keypoints_batch)
    keypoints_batch = to_numpy(keypoints_batch)
    imgs_out = []
    for i in range(len(imgs)):
        imgs_out.append(draw_coco17_skeleton(imgs[i], keypoints_batch[i], conf_thr))
    return imgs_out
