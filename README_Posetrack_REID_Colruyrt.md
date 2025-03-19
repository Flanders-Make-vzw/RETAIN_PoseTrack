# Colruyt: A Robust Online Multi-Camera People Tracking System With Geometric Consistency and State-aware Re-ID Correction

## TODO:

- [ ] Use process of Ewoud to extract bounding boxes with correct track ID for given cameras
- [ ] Evaluate number of times a poses can be extracted from a given bounding box => check a few examples
- [ ] Evaluate feature vector of identical people using original dataset of Colruyt
- [ ] Plug custom people detector of Colruyt and export to correct format
- [ ] Hyperparameter tuning

## Done: 

- [x] Extract images of people cropped to bounding box for camera
- [x] Evaluation of the results against those of Colruyt

This is the official repository for the winning submission to the 8th NVIDIA AI City Challenge (2024) Track 1: Multi-Camera People Tracking.
- Github: 
- Link to the paper: 


## Overall Pipeline

<img src="architecture.png" width="650" />

## Dataset

A restricted dataset of the 400 first frames can be downloaded from [sharepoint](https://flandersmake.sharepoint.com/:f:/s/ap_20220182RETAINVILICON-FM_channel/EkeAe2jzxNxNkwwHbz33JnUBZsAnC6A20hM5bhl3-gAaYQ?e=quuZo5). You need to copy the data into the test folder into a specific scene. 


The dataset should be placed in the `dataset` folder as follows:

```
dataset/
    test/
        scene_001/
            10_48_26_1/
                video.mp4
            10_48_26_10/
                video.mp4
            ...
        scene_002/
            cam_0001/
                video.mp4
            cam_0002/
                video.mp4
            ...
        ...
```

## Environment Setup

A docker container has been built for running the code with the dependencies.

### 1. Build the container

./docker/build_container.sh

### 2. Run the container

./docker/run_container.sh


## Inferencing
### 1. Prepare Pre-trained weights

Download the pretrained weights from [ByteTrack Yolox](https://drive.google.com/file/d/1LVFqYqx88R0TUjCMbTaKrJkL7-SdCSmC/view?usp=drive_link), [LUPerson Resnet](https://drive.google.com/file/d/1xDKWJRWja01nNOeV7TWcn58sHYSal2k9/view?usp=drive_link) and [Hrnet](https://drive.google.com/file/d/1tNT6gOBB95qYPCypvCctj1o-r7bzdwxA/view?usp=drive_link). Put all these pretrained weights in the `ckpt_weight` folder.

### 2. Detection
You can choose between the accelerated method detailed in step 2.1, which utilizes torch2trt, or the standard method described in step 2.2 if you prefer not to use it.

#### 2.1 Accelerated Detection (torch2trt)
```
python ./detection/utils/trt.py
script/fast_detection.sh
```
#### 2.2 Standard Detection (pytorch)

The standard script for detecting (includign a typo in the name) should be adapted Based on the actual hardware conditions, you can modify the values of `gpu_nums_per_iter`, `cpu_nums_per_iter`, and `scene_per_iter` in the shell script to achieve faster speeds or lower resource usage. In extreme cases, setting all three to 1 can accommodate the minimum hardware requirements. The same considerations regarding hardware configuration adjustments apply to the following scripts. 

```Bash
start = starting scene # scene_001 if start=1
end = ending scene #scene_020 if end = 20
scene_per_iter=1
gpu_nums_per_iter=1 # gpu_nums_per_iter >= 1
cpu_nums_per_item=4 #cpu_nums_per_item >= 1
```

```Bash
./script/standrad_detection.sh
```

If this does not work, just use the python function directly. Parameters need to be tuned!

**Note - REID inside detection does not seem to work ...**

```Bash
--with-reid --fast-reid-config fast_reid/configs/MOT17/sbs_S50.yml --fast-reid-weights pretrained/mot17_sbs_S50.pth
```

```Bash
python3 detection/get_detection.py --scene 2  --proximity_thresh 0.5 --appearance_thresh 0.25 --fuse --save_processed_img --save_annotated_img --batchsize 16
```

**Saving the annotated images:**
```Bash
python3 detection/get_detection.py --input /mnt/shared_disk/code_projects/Retain_asset_reID/PoseTrack/dataset/test/slim_people_tracking_data_entrance_checkout_first_isles --proximity_thresh 0.5 --appearance_thresh 0.25 --fuse --conf 0.1 --tsize 640 --track_high_thresh 0.5 --batchsize 16 --save_processed_img --save_annotated_img
```



### 3. Pose Estimation

```Bash
python3 mmpose/demo/save_pose_with_det_multiscene.py mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py ckpt_weight/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth --input dataset/test/scene_002 --output-root result/pose/scene_002 --draw-bbox --show-kpt-idx --start 1 --end 3
```

**For single camera pose estimation:**

```Bash
python3 mmpose/demo/save_pose_with_det_single_scene.py mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py ckpt_weight/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth --input dataset/test/slim_people_tracking_data_entrance_checkout_first_isles/20250115_tracked_10_48_26_28.mp4 --output-dir result/pose/ --show-kpt-idx --det-file dataset/test/slim_people_tracking_data_entrance_checkout_first_isles/20250115_tracked_10_48_26_28.txt
```

### 4. Re-ID

Download the weight from [ReID model](https://drive.google.com/file/d/17qbBmBX7DiT2lOuQ6rGHl8s9deKHkVn2/view?usp=sharing). Put this finetuned weight in the `ckpt_weight` folder.

Original script:
```Bash
script/reid_infer.sh
```

Adapted python script to function with correct offset of start/end scene
```Bash
cd fast-reid
python3 tools/infer.py --start 2 --end 2
```

**Single camera feature vector building:**



### 5. Tracking

Original SCript:
```
script/run_track.sh
```

Adapted Python script:
```Bash
python3 track/run_tracking.py --det_dir result/detection/slim_people_tracking_data_entrance_checkout_first_isles --pose_dir result/pose/slim_people_tracking_data_entrance_checkout_first_isles/poses --reid_dir result/reid/slim_people_tracking_data_entrance_checkout_first_isles --cal_file dataset/test/slim_people_tracking_data_entrance_checkout_first_isles/calibration.json --save_path result/track/slim_people_tracking_data_entrance_checkout_first_isles --verbose 
```