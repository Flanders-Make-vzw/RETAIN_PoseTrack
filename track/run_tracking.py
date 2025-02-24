import argparse
from pathlib import Path
import json
import numpy as np
from util.camera import Camera
from Tracker.PoseTracker import Detection_Sample, PoseTracker, TrackState
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_camera_calib(cal_file):
    """
    Load camera calibration data from a JSON file.

    Args:
        cal_file (Path): Path to the calibration JSON file.

    Returns:
        List[Camera]: List of Camera objects created from the calibration data.
    """
    with open(cal_file, 'r') as file:
        camera_calibs = json.load(file)

    cameras = []
    for cam_data in camera_calibs["calibration_info"]["cameras"]:
        camera = Camera.from_colruyt_calib(cam_data)
        cameras.append(camera)
    return cameras


def process_frame(tracker, frame_id, det_data, pose_data, reid_data, box_thred):
    """
    Process a single frame of data.

    Args:
        tracker (PoseTracker): The PoseTracker object.
        frame_id (int): The ID of the frame to process.
        det_data (List[np.ndarray]): List of detection data arrays.
        pose_data (List[np.ndarray]): List of pose data arrays.
        reid_data (List[np.ndarray]): List of reid data arrays.
        box_thred (float): The threshold for filtering detections.

    Returns:
        np.ndarray: The results for the processed frame.
    """
    detection_sample_mv = []
    for v in range(tracker.num_cam):
        detection_sample_sv = []
        det_sv = det_data[v]
        
        # Skip if no detections or empty array
        if det_sv is None or len(det_sv) == 0:
            detection_sample_mv.append(detection_sample_sv)
            continue
        
        # Convert to numpy array if not already
        det_sv = np.asarray(det_sv)
        
        idx = det_sv[:, 0] == frame_id
        cur_det = det_sv[idx]
        cur_pose = pose_data[v][idx]
        cur_reid = reid_data[v][idx]

        for det, pose, reid in zip(cur_det, cur_pose, cur_reid):
            if det[-1] < box_thred or len(det) == 0:
                continue
            new_sample = Detection_Sample(bbox=det[2:], keypoints_2d=pose[6:].reshape(17, 3), reid_feat=reid, cam_id=v, frame_id=frame_id)
            detection_sample_sv.append(new_sample)
        detection_sample_mv.append(detection_sample_sv)

    print("frame {}".format(frame_id), "det nums: ", [len(L) for L in detection_sample_mv])

    tracker.mv_update_wo_pred(detection_sample_mv, frame_id)

    frame_results = tracker.output(frame_id)
    return frame_results


def run_tracking(det_dir, pose_dir, reid_dir, cal_file, save_path):
    """
    Run the tracking process.

    Args:
        det_dir (Path): Directory for detection results.
        pose_dir (Path): Directory for pose results.
        reid_dir (Path): Directory for reid results.
        cal_file (Path): Calibration file for the cameras.
        save_path (Path): Path to save the tracking results.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print("exit", save_path)
        return

    cals = load_camera_calib(cal_file)

    det_data = []
    pose_data = []
    reid_data = []

    det_files = sorted(det_dir.glob('*.txt'))
    for det_file in det_files:
        base_name = det_file.stem
        pose_file = pose_dir / f"{base_name}.txt"
        reid_file = reid_dir / f"{base_name}.npy"

        if not pose_file.exists():
            logger.warning(f"Missing pose file {pose_file} for {base_name}")
        if not reid_file.exists():
            logger.warning(f"Missing file {reid_file} for {base_name}")

        try:
            det_file_data = np.loadtxt(det_file, delimiter=",")
            if det_file_data.ndim == 1:
                det_file_data = det_file_data.reshape(1, -1)
            det_data.append(det_file_data)

            # if the detection file is empty and the pose file does not exist, create an empty data.
            if not pose_file.exists():
                det_data.append(np.empty((0, 7)))
            
            # if the detection file is empty and the reid file does not exist, create an empty data.
            if not reid_file.exists() or reid_file.stat().st_size == 0:
                reid_data.append(np.empty((0, 128)))
            
            pose_file_data = np.empty((0, 18))
            if pose_file.exists():
                pose_file_data = np.loadtxt(pose_file, delimiter=",")
                if pose_file_data.ndim == 1:
                    pose_file_data = pose_file_data.reshape(1, -1)

            pose_data.append(pose_file_data)

            reid_file_data = np.empty((0, 128))
            if reid_file.exists():
                reid_file_data = np.load(reid_file, mmap_mode='r')
                if len(reid_file_data):
                    reid_file_data = reid_file_data / np.linalg.norm(reid_file_data, axis=1, keepdims=True)
            reid_data.append(reid_file_data)

        except Exception as e:
            logger.warning(f"Error loading data for {base_name}: {e}")
            continue
        
    if len(det_data) != len(pose_data) or len(det_data) != len(reid_data):
        logger.warning(f"Inconsistent data lengths for {base_name}: "
                        f"det={len(det_data)}, pose={len(pose_data)}, reid={len(reid_data)}")

    max_frame = []
    for det_sv in det_data:
        if det_sv.size == 0:
            continue

        print(det_sv.shape)
        print(det_sv)
        if len(det_sv):
            max_frame.append(np.max(det_sv[:, 0]))
    if not max_frame:
        print("No valid detection data found.")
        return
    max_frame = int(np.max(max_frame))

    tracker = PoseTracker(cals)
    box_thred = 0.3
    results = []

    for frame_id in tqdm(range(max_frame + 1), desc=det_dir.name):
        frame_results = process_frame(tracker, frame_id, det_data, pose_data, reid_data, box_thred)
        results += frame_results

    results = np.concatenate(results, axis=0)
    sort_idx = np.lexsort((results[:, 2], results[:, 0]))
    results = np.ascontiguousarray(results[sort_idx])

    np.savetxt(save_path, results)




if __name__ == '__main__':
    """
    Example of usage
    1. Run the script with the following command:
    
        python3 run_tracking.py --det_dir result/detection/scene_002 --pose_dir result/pose/scene_002 --reid_dir result/reid/scene_002 --cal_file data/people_tracking_data/calibration.json --save_path result/track/scene_002.txt
    """
    parser = argparse.ArgumentParser(description="Run tracking batch")
    parser.add_argument("--det_dir", type=Path, required=True, help="Directory for detection results")
    parser.add_argument("--pose_dir", type=Path, required=True, help="Directory for pose results")
    parser.add_argument("--reid_dir", type=Path, required=True, help="Directory for reid results")
    parser.add_argument("--cal_file", type=Path, required=True, help="Calibration file for the cameras")
    parser.add_argument("--save_path", type=Path, required=True, help="Path to save the tracking results")
    args = parser.parse_args()

    #extract the number of cameras from the detection directory

    run_tracking(
        det_dir=args.det_dir,
        pose_dir=args.pose_dir,
        reid_dir=args.reid_dir,
        cal_file=args.cal_file,
        save_path=args.save_path
    )