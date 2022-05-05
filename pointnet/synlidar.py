from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import yaml
import glob

class SynLiDARDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.cat = {}
        self.id2labels = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.num_classes = 0
        with open('synlidar.yaml', 'r') as stream:
            doc = yaml.safe_load(stream)
            self.id2labels = doc['labels']
        self.num_classes = len(self.id2labels)
        print("Number of classes: ", self.num_classes)
        files = glob.glob(root+'/*/velodyne/*.bin')
        print(len(files))
        self.cat = dict((v,k) for k,v in self.id2labels.items())
        self.id2cat = {v: k for k, v in self.cat.items()}
        self.meta = {}
        self.datapath = []
        for f_path in files:
            # scan = self.read_points(f_path)
            label_path = f_path.replace('velodyne', 'labels').replace('bin', 'label')
            self.datapath.append((f_path, label_path))
            # labels = self.read_label(label_path)
            # self.seg_labels = np.squeeze(np.eye(self.num_classes)[labels.reshape(-1)])
            # print(len(labels))
            # print(self.seg_labels[0])
            # print(self.seg_labels.shape)
            break
    def __getitem__(self, index: int):
        pt_path, label_path = self.datapath[index]
        scan = self.read_points(pt_path)
        labels = self.read_label(label_path)
        seg_labels = np.squeeze(np.eye(self.num_classes)[labels.reshape(-1)])
        print(scan.shape)
        print(seg_labels.shape)

        # random sampling
        # choice =  np.random.choice(len(seg_labels), self.npoints, replace=True)
        # scan = scan[choice]
        # seg_labels = seg_labels[choice]

        return scan, seg_labels
    def __len__(self):
        return len(self.datapath)



    def read_points(self, path):
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # [x,y,z,intensity]
        return scan

    def read_label(self, path):
        label = np.fromfile(path, dtype=np.uint32)
        label = label.reshape((-1))
        return label


if __name__ == '__main__':
    # dataset = sys.argv[1]
    # datapath = sys.argv[2]
    
    datapath = '../../dataset/SynLiDAR/sequences'

    d = SynLiDARDataset(root=datapath)
    len(d)
    d[0]

