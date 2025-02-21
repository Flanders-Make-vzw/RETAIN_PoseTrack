# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import time
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def infer_one_image(args,frame, bboxes_s, pose_estimator):
    #bboxes_s = bboxes_s[bboxes_s[:,4] > args.bbox_thr]
    pose_results = inference_topdown(pose_estimator, frame, bboxes_s[:,:4])
    records=[]
    for i, result in enumerate(pose_results):
        keypoints = result.pred_instances.keypoints[0]
        scores = result.pred_instances.keypoint_scores.T
        record = (np.concatenate((keypoints,scores),axis=1)).flatten()
        records.append(record)
    records = np.array(records)
    records = np.concatenate((bboxes_s,records),axis=1)
    return records
        


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--input', type=str, default='', help='Image/Video file')
    parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
    parser.add_argument('--output-root', type=str, default='', help='root of the output img file. Default not saving the visualization images.')
    parser.add_argument('--save-predictions', action='store_true', default=False, help='whether to save predicted results')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--det-cat-id', type=int, default=0, help='Category id for bounding box detection model')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.3, help='IoU threshold for bounding box NMS')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Visualizing keypoint thresholds')
    parser.add_argument('--draw-heatmap', action='store_true', default=False, help='Draw heatmap predicted by the model')
    parser.add_argument('--show-kpt-idx', action='store_true', default=False, help='Whether to show the index of keypoints')
    parser.add_argument('--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-3]
    root_path = '/'.join(path_arr)
    
    det_root = os.path.join(root_path, "result/detection")
    vid_root = os.path.join(root_path, "dataset/test")
    save_root = os.path.join(root_path, "result/pose")

    scenes = sorted(os.listdir(det_root))
    scenes = [s for s in scenes if s[0] == "s"]
    scenes = scenes[args.start:args.end]

    # print content of all variables to debug why the script is not processing all the files
    print(f"det_root: {det_root}")
    print(f"vid_root: {vid_root}")
    print(f"save_root: {save_root}")
    print(f"scenes: {scenes}")

    for scene in tqdm(scenes):
        print(f"Processing scene: {scene}")
        det_dir = os.path.join(det_root, scene)
        vid_dir = os.path.join(vid_root, scene)
        save_dir = os.path.join(save_root, scene)
        cams = os.listdir(vid_dir)
        print(f"cams: {cams}")

        # cams = sorted([c for c in cams if c.endswith(".txt")])
        print(f"Filtered cams: {cams}")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for cam in tqdm(cams):
            print(f"Processing cam: {cam}")
            det_path = os.path.join(det_dir, cam) + ".txt"
            vid_path = os.path.join(vid_dir, cam) + "/video.mp4"
            save_path = os.path.join(save_dir, cam + ".txt")
            print(f"det_path: {det_path}")
            print(f"vid_path: {vid_path}")
            print(f"save_path: {save_path}")

            if os.path.exists(save_path):
                print(f"Skipping {save_path}, already exists.")
                continue

            try:
                det_annot = np.loadtxt(det_path, delimiter=",")
            except Exception as e:
                print(f"Error reading detection file {det_path}: {e}")
                continue

            print(f"Loaded det_annot with shape: {det_annot.shape}")

            # Skip if no detections
            if det_annot.size == 0:
                print(f"No detections for {cam}, skipping.")
                continue

            # Ensure det_annot is 2-dimensional
            if det_annot.ndim == 1:
                det_annot = np.expand_dims(det_annot, axis=0)

            try:
                video = mmcv.VideoReader(vid_path)
            except Exception as e:
                print(f"Error reading video file {vid_path}: {e}")
                continue

            if len(video) == 0:
                print(f"Video file {vid_path} is empty, skipping.")
                continue

            all_results = []
            for frame_id, frame in enumerate(tqdm(video)):
                dets = det_annot[det_annot[:, 0] == frame_id]
                bboxes_s = dets[:, 2:7]  # x1y1x2y2s
                if len(bboxes_s) == 0:
                    continue

                result = infer_one_image(args, frame, bboxes_s, pose_estimator)
                result = np.concatenate((np.ones((len(result), 1)) * frame_id, result.astype(np.float32)), axis=1)
                all_results.append(result)

            if all_results:
                all_results = np.concatenate(all_results)
                np.savetxt(save_path, all_results)
                print(f"Saved results to {save_path}")
            else:
                print(f"No results to save for {cam}")

if __name__ == '__main__':
    main()
