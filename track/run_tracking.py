import argparse
from functools import partial
import os
import os.path as osp
from pathlib import Path
import time
import cv2
import json
import numpy as np
import sys
from util.camera import Camera
from Tracker.PoseTracker import Detection_Sample, PoseTracker,TrackState
from tqdm import tqdm
import copy

# json encoder class
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Camera):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


def run_tracking(det_dir, pose_dir, reid_dir, cal_file, save_path, verbose=False):
    """
    Run the tracking process for multi-view detections, poses, and re-identification features.

    Parameters:
        det_dir (str): Directory containing detection files.
        pose_dir (str): Directory containing pose files.
        reid_dir (str): Directory containing re-identification feature files.
        cal_file (str): Path to the camera calibration file.
        save_path (str): Path to save the tracking results.
        verbose (bool, optional): If True, print detailed information during processing. Default is False.

    Notes:
    - The function reads detection, pose, and re-identification data from the specified directories.
    - It handles empty and single-line detections by reshaping them appropriately.
    - It loads camera calibration data and initializes a PoseTracker object.
    - The tracking process is performed frame by frame, updating the tracker with multi-view detection samples.
    - The results are saved to the specified save_path.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    det_data=[]
    pose_data=[]
    reid_data = []

    cam_files = sorted(os.listdir(det_dir))
    sorted_cameras = {}
    cam_idx = 0

    num_detection_features = 7
    num_pose_features = 57
    num_reid_features = 2048

    if verbose:
        print("\nDETECTION (#detections x # detection features ({num_detection_features}) [frame_id, class, x1, y1, x2, y2, conf] )")
        print(f"\nPOSE (#detections x # pose features ({num_pose_features}) [bbox, conf, kpts (x,y,conf)] )")
        print(f"\nREID (#detections x #num reid features{num_reid_features})")

    cameras_to_ignore = []
    for cam_file in cam_files:
        cam_name = cam_file.replace(".txt", "").replace("_", ".")
        det_data_camera=np.loadtxt(osp.join(det_dir,cam_file), delimiter=",")
        # handle empty detection
        if len(det_data_camera)==0:
            print(f"WARNING: {cam_file} is empty, skipping camera.")
            cameras_to_ignore.append(cam_name)
            continue
        # handle single line detection
        elif len(det_data_camera.shape)==1:
            det_data_camera=det_data_camera.reshape(1,num_detection_features)

        # sorted
        sorted_cameras[cam_name] = cam_idx
        cam_idx += 1

        if verbose: print(f"\tDetection {cam_file}: {det_data_camera.shape}")
        det_data.append(det_data_camera)

        pose_data_camera=np.loadtxt(osp.join(pose_dir,cam_file))
        # handle empty detection
        if len(pose_data_camera)==0:
            pose_data_camera=np.zeros((1,num_pose_features))
        # handle single line detection
        elif len(pose_data_camera.shape)==1:
            pose_data_camera=pose_data_camera.reshape(1,num_pose_features)

        if verbose: print(f"\tPose {cam_file}: {pose_data_camera.shape}")
        pose_data.append(pose_data_camera)

        reid_data_scene=np.load(osp.join(reid_dir,cam_file.replace(".txt", ".npy")),mmap_mode='r')
        # handle empty file
        if len(reid_data_scene)==0:
            reid_data_scene=np.zeros((1, num_reid_features))
        # handle single line detection
        elif len(reid_data_scene.shape)==1:
            reid_data_scene=reid_data_scene.reshape(1, num_reid_features)

        if verbose: print(f"\tReid {cam_file}: {reid_data_scene.shape}")
        # normalize reid feature
        if len(reid_data_scene):
            reid_data_scene=reid_data_scene/np.linalg.norm(reid_data_scene, axis=1,keepdims=True)
        reid_data.append(reid_data_scene)

    max_frame = []
    for det_sv in det_data:
        if len(det_sv):
            max_frame.append(np.max(det_sv[:,0]))
    max_frame = int(np.max(max_frame))

    # sort cameras according to files
    # load camera calibration
    with open(cal_file, 'r') as file:
        data_camera_calibs = json.load(file)

    if verbose: (cameras_to_ignore)
    camera_calibs = {}
    for cam_data in data_camera_calibs["calibration_info"]["cameras"]:
        camera_name = cam_data["camera_serial"]
        if camera_name in cameras_to_ignore:
            continue
        cam_idx = sorted_cameras[camera_name]
        camera = Camera.from_colruyt_calib(cam_data, idx=cam_idx)
        camera_calibs[camera_name] = camera

    sorted_camera_calibs = [
        camera_calibs[name]
        for name in sorted(sorted_cameras, key=lambda x: sorted_cameras[x])]
    
    # save camera idx_int to json file
    save_path_json = save_path.replace(".txt", ".json")
    with open(save_path_json, 'w') as f:
        json.dump(sorted_camera_calibs, f, indent=4, cls=NpEncoder)


    tracker = PoseTracker(sorted_camera_calibs)

    box_thred = 0.1
    results = []

    for frame_id in tqdm(range(max_frame + 1), desc="Scene 002"):
        detection_sample_mv = []  # Multi-view detection samples

        # Process each camera
        for cam in tracker.cameras:
            if verbose: print(f"\t [{frame_id}/{max_frame}] - [{cam.idx_int + 1:02}/{len(tracker.cameras):02}] Processing camera {cam.idx_int:02} with serial {cam.camera_serial}")

            detection_sample_sv = []  # Single-view detection samples
            det_sv = det_data[cam.idx_int]  # Detection data for the current camera view

            assert len(det_sv) > 0, f"Detection data for this camera #{cam.idx_int} with serial {cam.camera_serial} is empty."

            idx = det_sv[:, 0] == frame_id  # Index for the current frame
            cur_det = det_sv[idx]  # Current detections for the current frame
            cur_pose = pose_data[cam.idx_int][idx]  # Current poses for the current frame
            cur_reid = reid_data[cam.idx_int][idx]  # Current reid features for the current frame

            for det, pose, reid in zip(cur_det, cur_pose, cur_reid):
                if det[-1] < box_thred or len(det) == 0:
                    continue

                bbox = det[2:]
                keypoints_2d = pose[6:].reshape(17, 3)
                reid_feat = reid
                cam_id = cam.idx_int
                frame_id = frame_id

                assert bbox.shape == (5,), f"Expected bbox shape (5,), got {bbox.shape}"
                assert keypoints_2d.shape == (17, 3), f"Expected keypoints_2d shape (17, 3), got {keypoints_2d.shape}"
                assert reid_feat.shape == (2048,), f"Expected reid_feat shape (2048,), got {reid_feat.shape}"

                new_sample = Detection_Sample(bbox=bbox, keypoints_2d=keypoints_2d, reid_feat=reid_feat, cam_id=cam_id, frame_id=frame_id)
                detection_sample_sv.append(new_sample)

            detection_sample_mv.append(detection_sample_sv)

            # print(f"[{cam.idx_int + 1:02}/{len(tracker.cameras):02}] camera {cam.idx_int:02} [{cam.camera_serial}] - #detections = {len(detection_sample_sv)}")
            assert cam.pos.shape == (3,) , f"Camera {cam.idx_int:02} with serial {cam.camera_serial} has wrong shape: Expected (3,) got {cam.pos.shape}"

        # TODO: add reduction of detections to cameras that overlap

        # Update tracker
        tracker.mv_update_wo_pred(detection_sample_mv, frame_id)

        frame_results = tracker.output(frame_id)
        results += frame_results

    results = np.concatenate(results,axis=0)
    sort_idx = np.lexsort((results[:,2],results[:,0]))
    results = np.ascontiguousarray(results[sort_idx])
    np.savetxt(save_path, results)


if __name__ == '__main__':
    """
    Example of usage
    1. Run the script with the following command:
    
        python3 track/run_tracking.py --det_dir result/detection/scene_002 --pose_dir result/pose/scene_002 --reid_dir result/reid/scene_002 --cal_file dataset/calibration.json --save_path result/track/scene_002.txt
    """
    parser = argparse.ArgumentParser(description="Run tracking batch")
    parser.add_argument("--det_dir", type=str, required=True, help="Directory for detection results")
    parser.add_argument("--pose_dir", type=str, required=True, help="Directory for pose results")
    parser.add_argument("--reid_dir", type=str, required=True, help="Directory for reid results")
    parser.add_argument("--cal_file", type=str, required=True, help="Calibration file for the cameras")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the tracking results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--cameras", type=str, nargs='+', help="List of cameras to process")
    args = parser.parse_args()

    run_tracking(
        det_dir=args.det_dir,
        pose_dir=args.pose_dir,
        reid_dir=args.reid_dir,
        cal_file=args.cal_file,
        save_path=args.save_path
    )