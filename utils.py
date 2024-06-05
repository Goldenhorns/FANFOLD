import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True 

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def split_tensor_by_ptr(x, ptr, readout='avg'):
    len_ptr= len(ptr)
    num_points = len_ptr - 1
    graph_set = torch.zeros(num_points, x.shape[1])
    for i in range(len_ptr-1):
        graph = x[ptr[i]:ptr[i+1]]
        if readout == 'avg':
            graph=torch.mean(graph, 0)
        elif readout == 'min':
            graph=torch.min(graph, 0).values
        elif readout == 'max':
            graph=torch.max(graph, 0).values
        graph_set[i]=graph
    return graph_set

def edge_index_to_adjacency_matrix(edge_index, num_nodes):  
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵  
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)  
      
    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上  
    adjacency_matrix[edge_index[0], edge_index[1]] = 1  
    adjacency_matrix[edge_index[1], edge_index[0]] = 1  
      
    return adjacency_matrix

def adj_to_image(A, name):
    A=A.detach().numpy()
    block_size = 20

    # 放大矩阵以便更清晰显示
    scaled_matrix = np.kron(A, np.ones((block_size, block_size), dtype=np.uint8))

    # 将矩阵转换为灰度图像
    image_array = scaled_matrix * 255  # 将 0 和 1 转换为灰度值范围内的值
    image = Image.fromarray(image_array.astype(np.uint8), 'L')

    # 保存灰度图像
    image.save('{}.png'.format(name))

def read_auc():
    with open(r'./reslut/ALL.txt', 'r', encoding='utf-8') as file:
        auc_map={}
        lines = file.readlines()
        for line in lines:
            line=line.split(',')
            line[1]=float(line[1].strip())
            line[2]=float(line[2].strip())
            auc_map[line[0].strip()]=line[1:]
    return auc_map

def save2excel(name):
    filename=name
    path=r"./Result".format(filename)
    