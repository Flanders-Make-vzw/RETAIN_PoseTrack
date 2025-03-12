# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from pathlib import Path

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
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Add new imports for visualization
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData

def infer_one_image(args, frame, bboxes_s, pose_estimator):
    #bboxes_s = bboxes_s[bboxes_s[:,4] > args.bbox_thr]
    pose_results = inference_topdown(pose_estimator, frame, bboxes_s[:,:4])
    records = []
    for i, result in enumerate(pose_results):
        keypoints = result.pred_instances.keypoints[0]
        scores = result.pred_instances.keypoint_scores.T
        record = (np.concatenate((keypoints, scores), axis=1)).flatten()
        records.append(record)
    records = np.array(records)
    records = np.concatenate((bboxes_s, records), axis=1)
    return records

def draw_bbox_on_image(image, bboxes, output_path, labels=None, 
                      color=(0, 255, 0), thickness=2, font_scale=0.5,
                      text_color=(255, 255, 255), text_thickness=1):
    """
    Draw bounding boxes on an image and save it.
    
    Args:
        image (numpy.ndarray): Input image in BGR format (OpenCV format)
        bboxes (numpy.ndarray): Array of bounding boxes in format [x1, y1, x2, y2, score]
                               or [x1, y1, x2, y2]
        output_path (str): Path where the annotated image will be saved
        labels (list, optional): List of labels for each bbox. Defaults to None.
        color (tuple, optional): Color of the bbox in BGR. Defaults to (0, 255, 0).
        thickness (int, optional): Thickness of the bbox lines. Defaults to 2.
        font_scale (float, optional): Font scale for text. Defaults to 0.5.
        text_color (tuple, optional): Color of the text in BGR. Defaults to (255, 255, 255).
        text_thickness (int, optional): Thickness of the text. Defaults to 1.
    
    Returns:
        numpy.ndarray: Annotated image
    """
    # Make a copy of the image to avoid modifying the original
    img_annotated = image.copy()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Draw each bounding box
    for i, bbox in enumerate(bboxes):
        # Extract coordinates
        if len(bbox) >= 5:  # If score is included
            x1, y1, x2, y2, score = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), bbox[4]
            score_text = f"{score:.2f}"
        else:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            score_text = ""
        
        # Draw rectangle
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if labels is not None and i < len(labels):
            if score_text:
                text = f"{labels[i]} {score_text}"
            else:
                text = f"{labels[i]}"
        else:
            text = score_text if score_text else ""
        
        # Draw label text if we have any
        if text:
            # Calculate text size to position the background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            
            # Draw text background
            cv2.rectangle(
                img_annotated,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                img_annotated,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                text_thickness
            )
    
    # Save the annotated image
    cv2.imwrite(output_path, img_annotated)
    
    return img_annotated

def main():
    """Process pose estimation for a single video/image using provided detections."""
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--input', type=str, required=True, help='Path to input video or image')
    parser.add_argument('--det-file', type=str, required=True, help='Path to detection file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--det-cat-id', type=int, default=0, help='Category id for bounding box detection model')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.3, help='IoU threshold for bounding box NMS')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Visualizing keypoint thresholds')
    parser.add_argument('--draw-heatmap', action='store_true', default=True, help='Draw heatmap predicted by the model')
    parser.add_argument('--show-kpt-idx', action='store_true', default=False, help='Whether to show the index of keypoints')
    parser.add_argument('--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--show', action='store_true', default=False, help='Whether to show visualization')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--draw-bbox', action='store_true', default=True, help='Whether to draw bounding boxes')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # Input validation
    assert os.path.exists(args.input), f"Input file does not exist: {args.input}"
    assert os.path.exists(args.det_file), f"Detection file does not exist: {args.det_file}"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the filename without extension for saving results
    input_filename = Path(args.input).stem
    output_file = os.path.join(args.output_dir, f"{input_filename}_pose.txt")
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}. Skipping.")
        return

    # Initialize visualizer for pose drawing
    visualizer = PoseLocalVisualizer(
        name='visualizer',
        radius=args.radius,
        line_width=args.thickness,
        # kpt_thr=args.kpt_thr,
        show_keypoint_weight=False,
        alpha=args.alpha
    )

    # Initialize pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # Load detection file
    try:
        det_annot = np.loadtxt(args.det_file, delimiter=",")
        print(f"Loaded detection file with shape: {det_annot.shape}")
    except Exception as e:
        print(f"Error loading detection file {args.det_file}: {e}")
        return

    # Skip if no detections
    if det_annot.size == 0:
        print(f"No detections in {args.det_file}, skipping.")
        return

    # Ensure det_annot is 2-dimensional
    if det_annot.ndim == 1:
        det_annot = np.expand_dims(det_annot, axis=0)

    # Load video or image
    is_image = Path(args.input).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    
    if is_image:
        # Handle single image
        frame = mmcv.imread(args.input)
        if frame is None:
            print(f"Error loading image file {args.input}")
            return
            
        # For images, assume all detections are for this frame
        bboxes_s = det_annot[:, 2:7]  # x1,y1,x2,y2,score
        
        if len(bboxes_s) == 0:
            print("No detections for this image.")
            return
            
        # Run pose estimation
        pose_results = inference_topdown(pose_estimator, frame, bboxes_s[:,:4])
        result = infer_one_image(args, frame, bboxes_s, pose_estimator)
        result = np.concatenate((np.zeros((len(result), 1)), result.astype(np.float32)), axis=1)  # Add frame_id=0
        np.savetxt(output_file, result)
        print(f"Saved results to {output_file}")
        
        # Create visualization using the visualizer
        data_sample = PoseDataSample()
        pred_instances = InstanceData()
    
        # Add detection and pose information to pred_instances
        if pose_results:  # Check if there are any pose results
            keypoints = np.concatenate([res.pred_instances.keypoints for res in pose_results])
            keypoint_scores = np.concatenate([res.pred_instances.keypoint_scores for res in pose_results])
            pred_instances.bboxes = bboxes_s[:, :4]
            pred_instances.keypoints = keypoints
            pred_instances.keypoint_scores = keypoint_scores
        else:
            print("No pose results to visualize.")
            # Create empty arrays to avoid errors
            pred_instances.bboxes = np.array([])
            pred_instances.keypoints = np.array([])
            pred_instances.keypoint_scores = np.array([])
    
        data_sample.pred_instances = pred_instances
    
        # Set dataset meta to visualizer so it knows about skeleton links
        if 'dataset_name' in pose_estimator.cfg:
            dataset_name = pose_estimator.cfg.dataset_name
        else:
            dataset_name = 'coco'  # Default to COCO dataset
        visualizer.set_dataset_meta(
            {'dataset_name': dataset_name}, 
            skeleton_style=args.skeleton_style
        )
    
        # Draw and save visualization
        vis_name = f"{input_filename}_pose_vis.png"
        vis_path = os.path.join(args.output_dir, vis_name)
    
        # Use visualizer to draw predictions
        drawn_img = visualizer.add_datasample(
            'pose_visualization',
            frame,
            data_sample,
            draw_gt=False,
            draw_pred=True,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=False,
            show=args.show,
            # kpt_thr=args.kpt_thr,
            out_file=vis_path
        )
    
        print(f"Saved visualization to {vis_path}")

    else:
        # Handle video file
        try:
            video = mmcv.VideoReader(args.input)
        except Exception as e:
            print(f"Error reading video file {args.input}: {e}")
            return
            
        if len(video) == 0:
            print(f"Video file {args.input} is empty, skipping.")
            return
            
        # Create video-specific output directory for frames
        video_frames_dir = os.path.join(args.output_dir, input_filename)
        os.makedirs(video_frames_dir, exist_ok=True)
            
        # Adjust start and end frames
        start_frame = max(0, args.start if hasattr(args, 'start') else 0)
        end_frame = args.end if hasattr(args, 'end') and args.end >= 0 else len(video) - 1
        end_frame = min(end_frame, len(video) - 1)
        
        all_results = []
        for frame_id in tqdm(range(start_frame, end_frame + 1), desc=f"Processing {input_filename}"):
            if frame_id >= len(video):
                break
                
            # Get frame
            frame = video[frame_id]
            
            # Get detections for this frame
            dets = det_annot[det_annot[:, 0] == frame_id]
            bboxes_s = dets[:, 2:7]  # x1,y1,x2,y2,score
            
            if len(bboxes_s) == 0:
                continue
                
            # Do pose estimation
            pose_results = inference_topdown(pose_estimator, frame, bboxes_s[:,:4])
            result = infer_one_image(args, frame, bboxes_s, pose_estimator)
            result = np.concatenate((np.ones((len(result), 1)) * frame_id, result.astype(np.float32)), axis=1)
            all_results.append(result)
            
            # Create visualization using the visualizer
            data_sample = PoseDataSample()
            pred_instances = InstanceData()
    
            # Add detection and pose information to pred_instances
            if pose_results:  # Check if there are any pose results
                keypoints = np.concatenate([res.pred_instances.keypoints for res in pose_results])
                keypoint_scores = np.concatenate([res.pred_instances.keypoint_scores for res in pose_results])
                pred_instances.bboxes = bboxes_s[:, :4]
                pred_instances.keypoints = keypoints
                pred_instances.keypoint_scores = keypoint_scores
            else:
                print("No pose results to visualize.")
                # Create empty arrays to avoid errors
                pred_instances.bboxes = np.array([])
                pred_instances.keypoints = np.array([])
                pred_instances.keypoint_scores = np.array([])
    
            data_sample.pred_instances = pred_instances
    
            # Set dataset meta to visualizer so it knows about skeleton links
            if 'dataset_name' in pose_estimator.cfg:
                dataset_name = pose_estimator.cfg.dataset_name
            else:
                dataset_name = 'coco'  # Default to COCO dataset
            visualizer.set_dataset_meta(
                {'dataset_name': dataset_name}, 
                skeleton_style=args.skeleton_style
            )
    
            # Draw and save visualization
            frame_vis_path = os.path.join(video_frames_dir, f"frame_{frame_id:04d}_pose_vis.png")
    
            # Use visualizer to draw predictions
            drawn_img = visualizer.add_datasample(
                'pose_visualization',
                frame,
                data_sample,
                draw_gt=False,
                draw_pred=True,
                draw_bbox=args.draw_bbox,
                show_kpt_idx=False,
                show=args.show,
                # kpt_thr=args.kpt_thr,
                out_file=frame_vis_path
            )

        if all_results:
            all_results = np.concatenate(all_results)
            np.savetxt(output_file, all_results)
            print(f"Saved results to {output_file}")
            print(f"Saved visualizations to {video_frames_dir}")
        else:
            print(f"No results to save for {input_filename}")


if __name__ == '__main__':
    main()