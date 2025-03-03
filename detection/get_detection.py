import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import json

from loguru import logger

sys.path.append('detection')

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from utils.timer import Timer
from tqdm import tqdm


def preproc(image, input_size, mean, std, swap=(0,3,1,2)):
    padded_img = np.full((len(image), *input_size, 3), 114, dtype=np.uint8)
    img = np.array(image)
    r = min(input_size[0] / img.shape[1], input_size[1] / img.shape[2])
    for i in range(img.shape[0]):
        resized_img = cv2.resize(
            cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB),
            (int(img[i].shape[1] * r), int(img[i].shape[0] * r)),
            interpolation=cv2.INTER_LINEAR
        )
        padded_img[i, : int(img.shape[1] * r), : int(img.shape[2] * r)] = resized_img
    # padded_img = padded_img[:,:, :, ::-1]
    padded_img = padded_img / np.float32(255.0)

    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
        
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("--scene", type=int, default=88,help='scene number')
    #parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default='detection/yolox/exps/example/mot/yolox_x_mix_det.py', type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default="ckpt_weight/bytetrack_x_mot17.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    parser.add_argument("--batchsize",default=1, type=int, help="batchsize")
    
    # New arguments for saving images
    parser.add_argument("--save_rescaled", action="store_true", help="Save rescaled images")
    parser.add_argument("--save_padded", action="store_true", help="Save padded images")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        batchsize=1
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.batchsize = batchsize
                
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        # self.rgb_means = np.tile(np.array((0.485, 0.456, 0.406)).reshape(1, 1, 1, -1),(batchsize,exp.test_size[0], exp.test_size[1],1))
        # self.std = np.tile(np.array((0.229, 0.224, 0.225)).reshape(1, 1, 1, -1),(batchsize,exp.test_size[0], exp.test_size[1],1))
        self.rgb_means = np.tile(np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 1, -1),(batchsize,exp.test_size[0], exp.test_size[1],1))
        self.std = np.tile(np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 1, -1),(batchsize,exp.test_size[0], exp.test_size[1],1))

    def inference(self, raw_img, img, ratio, timer):
        img_info = {"id": 0}

        height, width = img.shape[1:3]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = raw_img
        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).float().to(self.device, non_blocking=True)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args, scene):
    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-2]
    root_path = '/'.join(path_arr)

    input = osp.join(root_path, 'dataset/test', 'scene_0' + "{:02d}".format(scene))
    out_path = osp.join(root_path, 'result/detection', 'scene_0' + "{:02d}".format(scene))

    rescaled_path = osp.join(out_path, 'rescaled')
    padded_path = osp.join(out_path, 'padded')
    json_path = osp.join(out_path, 'ratios_and_paddings.json')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if args.save_rescaled and not os.path.exists(rescaled_path):
        os.makedirs(rescaled_path)
    if args.save_padded and not os.path.exists(padded_path):
        os.makedirs(padded_path)

    cameras = sorted(os.listdir(input))
    scale = min(800 / 1080, 1440 / 1920)
    # print("Predictor test size:", predictor.test_size)

    def preproc_worker(img):
        return preproc(img, predictor.test_size, predictor.rgb_means, predictor.std)

    batchsize = args.batchsize
    # print(cameras)
    ratios_and_paddings = {}
    for cam in cameras:
        if int(cam.split('_')[1]) < 0:
            continue
        frame_id = 0
        results = []
        # print(cam)
        video_path = osp.join(input, cam, 'video.mp4')
        cap = cv2.VideoCapture(video_path)
        timer = Timer()
        memory_bank = []
        id_bank = []
        carry_flag = False
        end_flag = False
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Processing {cam}")

        while cap.isOpened() and not end_flag:
            ret, frame = cap.read()
            if not ret:
                end_flag = True

            if not end_flag:
                memory_bank.append(frame)
                id_bank.append(frame_id)

            frame_id += 1
            pbar.update(1)

            if frame_id % 1000 == 0:
                logger.info('Processing cam {} frame {} ({:.2f} fps)'.format(cam, frame_id, 1. / max(1e-5, timer.average_time) * batchsize))

            if frame_id % batchsize == 0 or end_flag:
                if memory_bank:
                    img_data = memory_bank
                    id_data = np.array(id_bank)
                    memory_bank = []
                    id_bank = []
                    carry_flag = True
                else:
                    break
            else:
                carry_flag = False
                continue

            if carry_flag:
                # Detect objects
                img_preproc, ratio = preproc_worker(img_data)

                # Calculate padding
                pad_x = (predictor.test_size[1] - img_data[0].shape[1] * ratio) / 2
                pad_y = (predictor.test_size[0] - img_data[0].shape[0] * ratio) / 2

                # Store ratio and padding for the current camera
                if cam not in ratios_and_paddings:
                    ratios_and_paddings[cam] = {
                        "ratio": ratio,
                        "pad_x": pad_x,
                        "pad_y": pad_y,
                        "predictor_test_size": list(predictor.test_size),
                        "rgb_means":predictor.rgb_means.tolist(),
                        "std": predictor.std.tolist()
                    }

                outputs, img_info = predictor.inference(img_data, img_preproc, ratio, timer)

                # Save rescaled and padded images if requested
                for i in range(len(img_data)):
                    if args.save_rescaled:
                        rescaled_img = cv2.resize(img_data[i], (int(img_data[i].shape[1] * ratio), int(img_data[i].shape[0] * ratio)))
                        rescaled_img_path = osp.join(rescaled_path, f"{cam}_frame_{id_data[i]:04d}_rescaled.jpg")
                        cv2.imwrite(rescaled_img_path, rescaled_img)

                    if args.save_padded:
                        padded_img = img_preproc[i].transpose(1, 2, 0) * 255.0
                        padded_img = padded_img[:, :, ::-1].astype(np.uint8)
                        padded_img_path = osp.join(padded_path, f"{cam}_frame_{id_data[i]:04d}_padded.jpg")
                        cv2.imwrite(padded_img_path, padded_img)

                outputs = outputs
                for out_id in range(len(outputs)):
                    out_item = outputs[out_id]
                    detections = []
                    if out_item is not None:
                        detections = out_item[:, :7].cpu().numpy()
                        detections[:, :4] /= scale
                        detections = detections[detections[:, 4] > 0.1]

                    for det in detections:
                        x1, y1, x2, y2, score, _, _ = det
                        results.append([cam, id_data[out_id], 1, int(x1), int(y1), int(x2), int(y2), score])

                timer.toc()

        pbar.close()
        output_file = osp.join(out_path, cam + '.txt')
        with open(output_file, 'w') as f:
            for cam, frame_id, cls, x1, y1, x2, y2, score in results:
                f.write('{},{},{},{},{},{},{}\n'.format(frame_id, cls, x1, y1, x2, y2, score))

    # Save ratios and paddings to a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(ratios_and_paddings, json_file, indent=4)


def main(exp, args,scene):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda:0" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        print('using trt')
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = 'ckpt_weight/yolox_trt.pth'
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16,args.batchsize)
    current_time = time.localtime()
    
    image_demo(predictor, None, current_time, args,scene)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args,args.scene)
