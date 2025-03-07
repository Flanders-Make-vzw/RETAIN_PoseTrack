from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def load_poses(pose_path, bbox_scale=1, pose_scale=1):
    """
    Load poses from a text file.
    
    Args:
        pose_path (str): Path to the pose data text file.
        bbox_scale (float): Scaling factor for bounding boxes.
        pose_scale (float): Scaling factor for keypoints.
    ):
    Returns:
        dict: Dictionary with frame numbers as keys and lists of poses as values.
    """
    pose_data = defaultdict(list)
    
    # Track statistics for out of bounds coordinates
    total_poses = 0
    
    with open(pose_path, 'r') as f:
        for line in f:
            # structure the pose dict as 
            values = list(map(float, line.strip().split()))
            total_poses += 1

            # Extract frame number
            frame_number = int(values[0])

            pose = {}
            # Scale the bounding box
            x1, y1, x2, y2 = np.array(values[1:5]) * bbox_scale
                        
            pose["bbox"] = np.array([x1, y1, x2, y2])
            pose["bbox_conf"] = float(values[5])  # float
            
            # Process keypoints
            keypoints = np.array(values[6:]) * pose_scale
            num_keypoints = len(keypoints) // 3
            keypoints = keypoints.reshape(num_keypoints, 3)  # reshape to (num_keypoints, 3)
            
            
            pose["kpts"] = keypoints
            pose_data[frame_number].append(pose)
    
    
    return pose_data

def draw_pose(frame, pose, conf_threshold=0.1):
    """
    Draw a pose on a frame.
    
    Args:
        frame (np.ndarray): The video frame.
        pose (dict): The pose data dictionary with bbox, bbox_conf, and kpts.
        conf_threshold (float): Threshold for keypoint confidence.
    """
    # Extract keypoints from the pose dictionary
    keypoints = pose["kpts"]  # Shape: (num_keypoints, 3)
    
    # Define connections between keypoints for visualization (COCO format)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face and neck
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Define colors for different parts
    colors = {
        'face': (255, 0, 0),    # Blue
        'torso': (0, 255, 0),   # Green
        'arms': (0, 0, 255),    # Red
        'legs': (255, 255, 0)   # Cyan
    }
    
    # Draw each keypoint
    for i in range(keypoints.shape[0]):
        x, y, conf = keypoints[i]
        if conf > conf_threshold:
            # Draw keypoint as a circle
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # Optionally add keypoint labels
            # cv2.putText(frame, f"{i}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    line_thickness = 4
    # Draw connections between keypoints
    for connection in connections:
        idx1, idx2 = connection
        
        # Skip if indices are out of bounds
        if idx1 >= keypoints.shape[0] or idx2 >= keypoints.shape[0]:
            continue
            
        x1, y1, conf1 = keypoints[idx1]
        x2, y2, conf2 = keypoints[idx2]
        
        # Draw line if both keypoints have sufficient confidence
        if conf1 > conf_threshold and conf2 > conf_threshold:
            # Choose color based on body part
            if idx1 <= 4 or idx2 <= 4:  # Face and neck
                color = colors['face']
            elif idx1 <= 10 and idx2 <= 10:  # Arms
                color = colors['arms']
            else:  # Legs
                color = colors['legs']
                
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    
    # Format bbox data as: x1, y1, x2, y2, class, conf
    bbox_data = (int(pose["bbox"][0]), int(pose["bbox"][1]), int(pose["bbox"][2]), int(pose["bbox"][3]), 0, pose["bbox_conf"])
    # Draw the bounding box
    draw_bounding_box(frame, bbox_data, conf_threshold=conf_threshold, color= (255, 255, 0))

            
def draw_bounding_box(frame, bbox, conf_threshold=0.1, color=(0, 255, 0)):
    """
    Draw a bounding box on a frame.
    
    Args:
        frame (np.ndarray): The video frame.
        bbox (tuple): The bounding box coordinates (x1, y1, x2, y2).
    """
    x1, y1, x2, y2, _, conf = bbox
    if conf < conf_threshold: return
    # clip coordinates to stay within frame dimensions
    x1 = max(0, min(frame.shape[1]-1, x1))
    y1 = max(0, min(frame.shape[0]-1, y1))
    x2 = max(0, min(frame.shape[1]-1, x2))
    y2 = max(0, min(frame.shape[0]-1, y2))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def load_bounding_boxes(bbox_path, scale=1, pad_x=0, pad_y=0):
    """
    Load bounding boxes from a text file using NumPy and adjust for padding and rescaling.

    Args:
        bbox_path (str): Path to the bounding box text file.
        scale (float): Scaling factor used during preprocessing.
    Returns:
        dict: Dictionary with frame numbers as keys and lists of bounding boxes as values.
    """
    bbox_data = defaultdict(list)
    
    try:
        data = np.loadtxt(bbox_path, delimiter=',')
    except OSError:
        print(f"Error: Could not open file {bbox_path}")
        return bbox_data
    except ValueError:
        print(f"Error: File {bbox_path} is empty or has invalid format")
        return bbox_data

    # print(data)
    # Handle single line case
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    
    for row in data:
        frame_number, label, x1, y1, x2, y2, conf = row
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

        bbox_data[int(frame_number)].append((int(x1), int(y1), int(x2), int(y2), label, float(conf)))
        
    return bbox_data

def load_reid(reid_path):
    """Load .npy array from file."""
    reid_data = np.load(reid_path, allow_pickle=True)
    # display some statistics
    print("ReID data shape: ", reid_data.shape)
    print("ReID data type: ", reid_data.dtype)

    return reid_data


def load_track(track_path, video_dir_path, camera_calib_path, output_path=None, detection_goal_size=(800, 1440)):
    """Load and extract the images for each track into a seperate track folder.
    Args:
        track_path (str): Path to the track data text file.
        video_dir_path (str): Path to the directory containing video files
        camera_tracks (dict, optional): Dictionary to store camera calibration.
            - Each camera has a camera idx and IP serial (corresponding to the video file name).
            Ex: 10.48.26.1 corresponds to {video_dir_path}/10_48_26_1/video.mp4
            
    Returns:
        None
        
    
    Each track (.txt) contains 
    - Camera_id
    - Track id 
    - Frame_id 
    - Bounding box: (x, y, width, height) 
    - 3D coordinates: (x, y, z)
    For example: 1.000000000000000000e+00 1.000000000000000000e+00 3.000000000000000000e+00 2.760000000000000000e+02 2.700000000000000000e+02 9.000000000000000000e+01 2.310000000000000000e+02 3.210000000000000000e+02 5.010000000000000000e+02 1.000000000000000000e+00
    Corresponds to:
    - Camera_id: 1
    - Track id: 1
    - Frame_id: 3
    - bbox: (276, 270, 90, 231)
    - Track.output_cord: (321, 501, 1)
    """

    goal_height, goal_width = detection_goal_size

    # Create output directory if provided
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Load track data from file
    try:
        track_data = np.loadtxt(track_path)
    except Exception as e:
        print(f"Error loading track file {track_path}: {e}")
        return {}
    
    # Handle single line case
    if track_data.ndim == 1:
        track_data = track_data.reshape(1, -1)
    
    # Group tracks by track ID
    tracks_by_id = defaultdict(list)
    for row in track_data:
        
        camera_id = int(row[0])
        track_id = int(row[1])
        frame_id = int(row[2])
        bbox = row[3:7]  # x, y, width, height
        pos_3d = row[7:10]  # x, y, z
        print("Camera ID: ", camera_id)
        print("Track ID: ", track_id)
        print("Frame ID: ", frame_id)
        print("Bounding box: ", bbox)
        print("3D position: ", pos_3d)
        
        tracks_by_id[track_id].append({
            'camera_id': camera_id,
            'frame_id': frame_id,
            'bbox': bbox,
            'pos_3d': pos_3d
        })
    # print("Tracks by ID: ", tracks_by_id)
    print("Number of tracks: ", len(tracks_by_id))
    for track_id, frames in tracks_by_id.items():
        print(f"Track ID: {track_id}, Number of frames: {len(frames)}")
    
    try:
        with open(camera_calib_path, 'r') as f:
            camera_calibs = json.load(f)
    except Exception as e:  
        print(f"Error loading camera calibration file {camera_calib_path}: {e}")
        return tracks_by_id
    
    # Extract frames for each track if output_path is provided
    if output_path:
        for track_id, frames in tracks_by_id.items():
            track_dir = output_path / f"track_{track_id}"
            track_dir.mkdir(exist_ok=True)
            
            # Group by camera
            by_camera = {}
            for frame in frames:
                camera_id = frame['camera_id']
                if camera_id not in by_camera:
                    by_camera[camera_id] = []
                by_camera[camera_id].append(frame)
            
            # Process each camera
            for camera_id, camera_frames in by_camera.items():
                # Find camera serial from calibration data
                camera_serial = None
                for cam_info in camera_calibs:
                    if cam_info.get('idx') == camera_id -1:  # Index might be 0-based
                        camera_serial = cam_info.get('camera_serial')
                        break
                
                if not camera_serial:
                    print(f"Warning: Camera ID {camera_id} not found in calibration data")
                    continue
                
                # Convert serial number format: 10.48.26.1 -> 10_48_26_1
                camera_folder = camera_serial.replace('.', '_')
                video_path = video_dir_path / camera_folder / "video.mp4"
                
                if not video_path.exists():
                    print(f"Warning: Video file not found at {video_path}")
                    continue
                
                # Open video and extract frames
                cap = cv2.VideoCapture(str(video_path))
                # get video size
                input_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                video_width = input_size[1]
                video_height = input_size[0]
                bbox_scale = min(video_height/ goal_height, video_width/goal_width)

                assert cap.isOpened(), f"Error: Could not open video {video_path}"
                
                for frame in sorted(camera_frames, key=lambda x: x['frame_id']):
                    frame_id = frame['frame_id']
                    bbox = frame['bbox']
                    
                    # Set video position to frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, img = cap.read()
                    
                    if not ret:
                        print(f"Warning: Could not read frame {frame_id} from video {video_path}")
                        continue
                    
                    # Draw bounding box
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Save frame
                    output_file = track_dir / f"{camera_serial}_frame{frame_id}.jpg"
                    cv2.imwrite(str(output_file), img)
                
                cap.release()
    
    return tracks_by_id

def annotate_detection(video_path, bbox_path, pose_path=None, show_video=False, detection_goal_size=(800, 1440), output_path=None, conf_threshold=0.1):
    """
    Annotate bounding boxes on video frames.
    Args:
        video_path (str): Path to the video file.
        bbox_path (str): Path to the bounding box txt file.
        pose_path (str): Path to the pose data txt file.
        show_video (bool): Whether to display the video with annotations.
        output_path (str): Path to save the annotated video.
    """

    # Load the video
    video = cv2.VideoCapture(video_path)

    # get video size
    input_size = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Video dims: (height x width): ", input_size)
    video_width = input_size[1]
    video_height = input_size[0]
    goal_height, goal_width = detection_goal_size

    # Calculate scaling factor correctly (matching get_detection.py)
    bbox_scale = min(video_height/ goal_height, video_width/goal_width)

    FPS = video.get(cv2.CAP_PROP_FPS)
    
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cv_out = cv2.VideoWriter(output_path, fourcc, FPS, (input_size[1], input_size[0]))

    # Load the bounding box dataset

    bbox_data = load_bounding_boxes(
        bbox_path,
        scale=bbox_scale
    )

    wait_key = 1
    
    # Load the pose dataset if provided
    pose_data = load_poses(
        pose_path, 
        bbox_scale=bbox_scale,
        pose_scale=bbox_scale,
    ) if pose_path else defaultdict(list)

    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    paused = False
    frame_number = 0

    while frame_number < total_frames:
        if not paused:
            ret, frame = video.read()
            frame_number += 1
            if not ret:
                break

            # Check if there are any bounding boxes for this frame
            if frame_number in bbox_data:
                for bbox in bbox_data[frame_number]:
                    draw_bounding_box(frame, bbox, conf_threshold=conf_threshold, color=(0, 255, 0))

            # Check if there are any poses for this frame
            if frame_number in pose_data:
                for pose in pose_data[frame_number]:
                    # Check if pose is out of bounds
                    draw_pose(frame, pose, conf_threshold=conf_threshold)

            # Display the frame with bounding boxes if show_video is True
            if show_video:
                cv2.imshow('Annotated Frame', frame)

            # Write the frame to the output video if output_path is provided
            if output_path is not None:
                cv_out.write(frame)

        # Handle key press events
        key = cv2.waitKey(wait_key) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
    
    # Release the video and close all OpenCV windows
    video.release()
    if output_path is not None:
        cv_out.release()
    if show_video:
        cv2.destroyAllWindows()

def annotate_results(show_video=False):
    scenes = ["scene_002"]
    dataset_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/dataset/test/")
    bbox_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/detection")
    annotated_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/annotated")
    pose_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/pose")
    reid_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/reid")
    track_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/track")


    for scene in dataset_path.iterdir():
        if not scene.is_dir(): continue
        if scene.name not in scenes: continue

        print("Scene: ", scene.name)
        for camera_dir in scene.iterdir():
            if not camera_dir.is_dir(): continue
            print("\t=> Camera: ", camera_dir.name)
            
            # check if output dir exists and create if not
            if not (annotated_results_path / scene.name).exists():
                (annotated_results_path / scene.name).mkdir(parents=True, exist_ok=True)
            ## iterate through videos
            for video_path in camera_dir.iterdir():
                if not video_path.is_file() or video_path.suffix != ".mp4": continue
                ## annotate bbox
                camera_txt = camera_dir.name + ".txt"
                camera_mp4 = camera_dir.name + ".mp4"
                
                bbox_path = bbox_results_path / scene.name / camera_txt
                if not bbox_path.exists(): continue

                pose_path = pose_results_path / scene.name / camera_txt
                if not pose_path.exists(): pose_path = None
                reid_path  = reid_results_path / scene.name / camera_txt
                if not reid_path.exists(): reid_path = None
                annotated_path = annotated_results_path / scene.name / camera_mp4
                if annotated_path.exists(): continue

                print("\t\t=> Video: ", video_path.name)
                print("\t\t=> BBox: ", bbox_path)
                print("\t\t=> Pose: ", pose_path)
                print("\t\t=> ReID: ", reid_path)
                print("\t\t=> Annotated: ", annotated_path)

                if not bbox_path.exists(): continue

                annotate_detection(
                    video_path=video_path,
                    bbox_path=bbox_path,
                    pose_path=pose_path,
                    output_path=annotated_path,
                    # reid_path=reid_path,
                    show_video=show_video)

def debug_annotation(show_video=False):
    # show_video = True
    scene = "scene_001"
    camera = "cam_0001"
    
    scene = "scene_002"
    camera = "10_48_26_2"
    video_path=None
    bbox_path=None
    pose_path=None
    reid_path=None
    
    
    dataset_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/dataset/test/")
    bbox_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/detection")
    pose_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/pose")
    reid_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/reid")
    track_results_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/track")
    
    video_path=dataset_path / scene / camera / "video.mp4"
    bbox_path=bbox_results_path / scene / (camera + ".txt")
    pose_path=pose_results_path / scene / (camera + ".txt")
    reid_path=reid_results_path / scene / (camera + ".npy")
    track_path=track_results_path / (scene + ".txt")
    camera_calib_path=track_results_path / (scene + ".json")
    track_images_output_path = Path("/mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/result/track_images/" + scene)
    print("Track output:", track_images_output_path)

    detection_goal_size = (800, 1440)

    # load reid => feature vector of identified individuals
    # reid = load_reid(reid_path)
    
    # load 
    track = load_track(
        track_path=track_path, 
        video_dir_path=dataset_path / scene, 
        camera_calib_path=camera_calib_path,
        output_path=track_images_output_path
        )

    # annotate_detection(
    #     video_path=video_path,
    #     bbox_path=bbox_path,
    #     # pose_path=pose_path,
    #     reid_path=reid_path,
    #     show_video=show_video,
    #     detection_goal_size=detection_goal_size
    # )


if __name__ == "__main__":
    debug_annotation(show_video=True)
