"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
import math as math
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

CAMERA_INFO = {
    'angel_ground': np.pi / 3,
    'height': 300,
}


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def get_moving_boxes(prev_image, image):
    frame_lwpCV = image
    background = prev_image
    # 对帧进行预处理，先转灰度图，再进行高斯滤波。
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        return None
    # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

    # 显示矩形框
    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 1500:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        boxes.append((x, y, w, h))
    return boxes


def get_out_boxes_velocities(prev_info, curr_info, threshold, interval, image_size):
    curr_boxes, curr_classes = curr_info
    velocities = []
    for i in range(len(curr_boxes)):
        velocities.append(get_moving_speed(prev_info, curr_boxes[i], curr_classes[i], threshold, interval, image_size))
    return velocities


def get_moving_speed(prev_info, curr_box, curr_class, threshold, interval, image_size):
    prev_boxes, prev_classes = prev_info
    closest_moving_box = None
    similarity = 9999

    def get_rec_similarity(rec1, rec2):
        (x1, y1, w1, h1) = rec1
        (x2, y2, w2, h2) = rec2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    for i in range(len(prev_boxes)):
        ts = get_rec_similarity(prev_boxes[i], curr_box)
        if ts < similarity and curr_class == prev_classes[i]:
            similarity = ts
            closest_moving_box = prev_boxes[i]

    if similarity < threshold:
        prev_center = get_center(closest_moving_box, image_size)
        curr_center = get_center(curr_box, image_size)
        angel_ground = CAMERA_INFO['angel_ground']
        horizontal_dis = curr_center[0] - prev_center[0]
        vertical_dis = (curr_center[1] - prev_center[1]) / math.cos(angel_ground)
        print("horizontal_dis: {}, vertical_dis: {}".format(horizontal_dis, vertical_dis))
        dis = np.sqrt(np.square(horizontal_dis) + np.square(vertical_dis))
        speed = dis / interval
        direction_x, direction_y = horizontal_dis / dis, vertical_dis / dis
        return speed, (direction_x, direction_y)
    return 0, (0, 0)


def get_center(box, image_size):
    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image_size[0], np.floor(right + 0.5).astype('int32'))
    return left + (right - left) / 2, top + (bottom - top) / 2