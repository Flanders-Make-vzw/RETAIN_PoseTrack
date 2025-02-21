#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

cd mmpose

set -x

start=0
end=1
scene_per_iter=5   #scene_per_iter={1,2,5,10,15,30}

for ((j=0; j < ($end-$start+1) / $scene_per_iter; j++)); do
  # Use a for loop to iterate through scenes
  for ((i = $start + $j * $scene_per_iter; i < $start + $j * $scene_per_iter + $scene_per_iter; i++)); do
      /usr/bin/python3.9 demo/save_pose_with_det_multiscene.py \
      demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
      https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco \
      configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
      ../ckpt_weight/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
      --input examples/88.jpg \
      --output-root /home/result/pose/ \
      --draw-bbox \
      --show-kpt-idx \
      --start $i \
      --end $((i + 1)) &
  done
  wait
done