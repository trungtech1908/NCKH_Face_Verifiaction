# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午2:13
# @Author : zhuying
# @Company : Minivision
# @File : utility.py
# @Software : PyCharm

from datetime import datetime
import os


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

import cv2

def get_crop_face(image, box, scale):
    x1, y1, x2, y2 = map(int, box)
    w = x2 - x1
    h = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Tạo khung vuông dựa trên cạnh lớn nhất
    side = max(w, h) * scale
    
    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(image.shape[1], int(cx + side / 2))
    ny2 = min(image.shape[0], int(cy + side / 2))
    
    return image[ny1:ny2, nx1:nx2]