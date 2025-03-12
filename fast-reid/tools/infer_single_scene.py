import numpy as np
from collections import deque
import os
import os.path as osp
import copy
torch
import torch.nn.functional as F
import argparse
import sys

import time
import cv2
import json
import torch
import yaml
import mmcv
from tqdm import tqdm
from torchvision.ops import roi_align, nms
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
path_arr = current_file_path.split('/')[:-3]
root_path = '/'.join(path_arr)
sys.path.append(os.path.join(root_path, 'fast-reid'))

class reid_inferencer():
    def __init__(self, reid):
        self.reid = reid
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
        self.device = self.reid.device

    def mgn(self, crops):
        features = self.reid.backbone(crops)  # (bs, 2048, 16, 8)
        b1_feat = self.reid.b1(features)
        b2_feat = self.reid.b2(features)
        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)
        b3_feat = self.reid.b3(features)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)

        b1_pool_feat = self.reid.b1_head(b1_feat)
        b2_pool_feat = self.reid.b2_head(b2_feat)
        b21_pool_feat = self.reid.b21_head(b21_feat)
        b22_pool_feat = self.reid.b22_head(b22_feat)
        b3_pool_feat = self.reid.b3_head(b3_feat)
        b31_pool_feat = self.reid.b31_head(b31_feat)
        b32_pool_feat = self.reid.b32_head(b32_feat)
        b33_pool_feat = self.reid.b33_head(b33_feat)

        pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat,
                               b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
        return pred_feat

    def process_frame(self, frame, bboxes):
        frame = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).cuda()
        frame = frame / 255.0
        paddingframe = torch.ones((3, 2160, 3840)).cuda()
        paddingframe[0] = 0.485
        paddingframe[1] = 0.456
        paddingframe[2] = 0.406

        paddingframe[:, 540:1620, 960:2880] = frame
        paddingframe.sub_(self.mean).div_(self.std)
        frame = paddingframe.unsqueeze(0)
        cbboxes = bboxes.copy()
        cbboxes[:, [1, 3]] += 540
        cbboxes[:, [0, 2]] += 960
        cbboxes = cbboxes.astype(np.float32)
        newcrops = roi_align(frame, torch.cat([torch.zeros(len(cbboxes), 1).cuda(), torch.from_numpy(cbboxes).cuda()], 1), (384, 128)).to(self.device)
        newfeats = (self.mgn(newcrops) + self.mgn(newcrops.flip(3))).detach().cpu().numpy() / 2

        return newfeats

    def process_frame_simplified(self, frame, bboxes):
        frame = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).cuda()
        frame = frame / 255.0
        frame.sub_(self.mean).div_(self.std)
        frame = frame.unsqueeze(0)
        cbboxes = bboxes.copy()
        cbboxes = cbboxes.astype(np.float32)
        newcrops = roi_align(frame, torch.cat([torch.zeros(len(cbboxes), 1).cuda(), torch.from_numpy(cbboxes).cuda()], 1), (384, 128)).to(self.device)
        newfeats = (self.mgn(newcrops) + self.mgn(newcrops.flip(3))).detach().cpu().numpy() / 2

        return newfeats

def main():
    parser = ArgumentParser()
    parser.add_argument('--det_path', type=str, help='Path to detection file')
    parser.add_argument('--vid_path', type=str, help='Path to video file')
    parser.add_argument('--save_path', type=str, help='Path to save file')
    args = parser.parse_args()

    reid = torch.load('../ckpt_weight/aic24.pkl').cuda().eval()
    reid_model = reid_inferencer(reid)

    det_path = args.det_path
    vid_path = args.vid_path
    save_path = args.save_path
    
    det_annot = np.ascontiguousarray(np.loadtxt(det_path, delimiter=","))
    if len(det_annot) == 0:
        all_results = np.array([])
        np.save(save_path, all_results)
    
    if det_annot.ndim == 1:
        det_annot = np.expand_dims(det_annot, axis=0)
    
    video = mmcv.VideoReader(vid_path)
    all_results = []
    
    for frame_id, frame in enumerate(tqdm(video)):
        dets = det_annot[det_annot[:, 0] == frame_id]
        bboxes_s = dets[:, 2:7]  # x1y1x2y2s
        
        screen_width = 1920
        screen_height = 1080
        
        x1 = bboxes_s[:, 0]
        y1 = bboxes_s[:, 1]
        x2 = bboxes_s[:, 2]
        y2 = bboxes_s[:, 3]
        
        x1 = np.maximum(0, x1)
        y1 = np.maximum(0, y1)
        x2 = np.minimum(screen_width, x2)
        y2 = np.minimum(screen_height, y2)
        
        bboxes_s[:, 0] = x1
        bboxes_s[:, 1] = y1
        bboxes_s[:, 2] = x2
        bboxes_s[:, 3] = y2
        
        if len(bboxes_s) == 0:
            continue
        with torch.no_grad():
            feat_sim = reid_model.process_frame_simplified(frame, bboxes_s[:, :-1])
        all_results.append(feat_sim)
    if all_results:
        all_results = np.concatenate(all_results)
    else:
        all_results = np.array([])
    np.save(save_path, all_results)

if __name__ == '__main__':
    main()