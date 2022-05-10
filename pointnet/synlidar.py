from __future__ import print_function
from collections import defaultdict
from posixpath import splitext
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
                 npoints=6000,
                 classification=False,
                 split='train',
                 data_augmentation=False):
        self.npoints = npoints
        self.root = root
        self.cat = {}
        self.id2labels = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.num_classes = 0
        self.sequences = {}
        self.splittype = split
        with open(os.path.join(self.root,'../synlidar.yaml'), 'r') as stream:
            doc = yaml.safe_load(stream)
            self.id2labels = doc['labels']
            self.split = doc['split']
        self.num_classes = len(self.id2labels)
        print("Number of classes: ", self.num_classes)
        # files = glob.glob(root+'/*/velodyne/*.bin')
        # print(len(files))
        self.cat = dict((v,k) for k,v in self.id2labels.items())
        self.id2cat = {v: k for k, v in self.cat.items()}
        self.meta = {}
        self.datapath = []
        # datapath: {[train/valid]:[(pt_path, label_path)]}
        # print(self.split)
        for num in self.split[self.splittype]:
            files = glob.glob(root+'/'+str(num)+'/velodyne/*.bin')
            # print("files:",files)
            for f_path in files:
                label_path = f_path.replace('velodyne', 'labels').replace('bin', 'label')
                # print(type(split_type))
                self.datapath.append((f_path, label_path))
                # labels = self.read_label(label_path)
                # self.seg_labels = np.squeeze(np.eye(self.num_classes)[labels.reshape(-1)])
                # print(len(labels))
                # print(self.seg_labels[0])
                # print(self.seg_labels.shape)
        print("Number of instances:", len(self.datapath))
        print("Dataset type:", self.splittype)


    def __getitem__(self, index: int):
        pt_path, label_path = self.datapath[index]
        scan = self.read_points(pt_path)
        labels = self.read_label(label_path)
        seg_labels = np.squeeze(np.eye(self.num_classes)[labels.reshape(-1)])
        # downsampling
        choice = np.random.choice(len(scan), self.npoints, replace=True)
        scan = scan[choice, :3]
        seg_labels = seg_labels[choice]
        labels = labels[choice].astype(np.int64)

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            scan[:,[0,2]] = scan[:,[0,2]].dot(rotation_matrix) # random rotation
            scan += np.random.normal(0, 0.02, size=scan.shape) # random jitter
        # random sampling
        # choice =  np.random.choice(len(seg_labels), self.npoints, replace=True)
        # scan = scan[choice]
        # seg_labels = seg_labels[choice]
        scan = torch.from_numpy(scan)
        seg_labels = torch.from_numpy(seg_labels)
        labels = torch.from_numpy(labels)
        return scan, labels
        
    def __len__(self):
        return self.npoints



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
    print(len(d[0][0]))
    print(len(d[1][0]))

