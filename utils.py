#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/10/7 11:13
# @File    : utils.py
# @Software: PyCharm
"""
import os
import time

import torch


def create_tensor(tensor):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        # device = torch.device("cuda:[0, 1]" if torch.cuda.is_available() else "cpu")
        tensor = tensor.cuda()
    return tensor



tensor = torch.randn(1000, 1000)

tensor_gpu = create_tensor(tensor)

start = time.time()

t1 = torch.mm(tensor, tensor)

end = time.time()

print(end-start)

start = time.time()

t1 = torch.mm(tensor_gpu, tensor_gpu)

end = time.time()

print(end-start)

start = time.time()

t1 = torch.mm(tensor_gpu, tensor_gpu)

end = time.time()

print(end-start)



