import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import _init_path
import argparse
import datetime
import glob
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from pcdet.models import load_data_to_gpu
import open3d as o3d
from open3d.core import Tensor, concatenate
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
import copy
import cv2
from pcdet.datasets.waymo.waymo_dataset import WaymoDataset

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    # parser.add_argument('--cfg_file', type=str, default='tools/cfgs/waymo_models/centerpoint.yaml', help='specify the config for training')
    parser.add_argument('--cfg_file', type=str, default='out_dir/waymo_models/vit_centerpoint_ver5/20230127/vit_centerpoint_ver5.yaml', help='specify the config for training')
    #parser.add_argument('--cfg_file', type=str, default='out_dir/waymo_models/centerbaselstm/20230124/centerbaselstm.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='Final_Results', help='extra tag for this experiment')
    # parser.add_argument('--ckpt', type=str, default='/media/WD6THDD/Detector/pretrained_model/centerpoint/default/ckpt/checkpoint_epoch_50.pth', help='checkpoint to start from')
    parser.add_argument('--ckpt', type=str, default='/media/WD6THDD/Detector/out_dir/waymo_models/vit_centerpoint_ver5/20230127/ckpt/checkpoint_epoch_10.pth', help='checkpoint to start from')
    #parser.add_argument('--ckpt', type=str, default='/media/WD6THDD/Detector/out_dir/waymo_models/centerbaselstm/centerbaselstm/20230124/ckpt/checkpoint_epoch_10.pth', help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--visualize', default=True, help='use visualizer')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='eval', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[2:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def custom_batch(batch_dict):
    for key, val in batch_dict.items():
        if key == 'points' or key == 'voxels' or key == 'split_length' or key == 'voxel_num_points' or key == 'voxel_coords':
            continue
        batch_dict[key] = np.array(val)
    batch_dict['batch_size'] = 1
    
update = True
index = 0
def visualize(model,test_loader, args, logger,WD, dist_test=False):
    global update,index
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    model.eval()

    vis = o3d.visualization.VisualizerWithKeyCallback()  # <-コールバック対応クラス
    vis.create_window(
        window_name="Hoge",  # ウインドウ名
        width=800,           # 幅
        height=600,          # 高さ
        left=50,             # 表示位置(左)
        top=50               # 表示位置(上)
    )  
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    def next_callback(_vis):
        global update,index
        update = True
        index = (index +  1) % len(WD)
        print('{:06d}/{:06d}'.format(index,len(WD)-1))
    vis.register_key_callback(262, next_callback)
    def back_callback(_vis):
        global update,index
        update = True
        index = (index + len(WD) - 1) % len(WD)
        print('{:06d}/{:06d}'.format(index,len(WD)-1))
    vis.register_key_callback(263, back_callback)
    def capture_callback(_vis):
        global index
        buffer = _vis.capture_screen_float_buffer(True)
        img = cv2.resize(np.asarray(buffer),(400,300))*255
        img = img[:,:,::-1]
        cv2.imwrite('capture/{:06d}.png'.format(index),img)
    vis.register_key_callback(ord('C'), capture_callback)
    def capture_image_for_video(_vis, image, imlist):
        buffer = _vis.capture_screen_float_buffer(True)
        img = cv2.resize(np.asarray(buffer),(400,300))*255
        imlist.append(image)
        
    #dataloader_iter = WD.__iter__()
    current_cloud = o3d.geometry.PointCloud()
    #is.add_geometry(current_cloud)
    ctr = vis.get_view_control()

    #param = o3d.io.read_pinhole_camera_parameters('ScreenCamera_2023-02-03-00-26-31.json')
    #ctr.convert_from_pinhole_camera_parameters(param)
    first = True
    capture_list = []
    while True:
        if update:
            vis.clear_geometries()
            batch_dict = WD.__getitem__(index)
            batch_dict = WD.collate_batch([batch_dict])
            custom_batch(batch_dict)
            #batch_dict = next(dataloader_iter)
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
            points = batch_dict['points'][0][:, 1:4].cpu().detach().numpy()
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().detach().numpy()
            gt_boxes = batch_dict['gt_boxes'].to('cpu').detach().numpy().copy()[0]

            for gt_box in gt_boxes:
                obb = o3d.geometry.OrientedBoundingBox(
                    gt_box[0:3],
                    np.array([[np.cos(gt_box[6]),np.sin(gt_box[6]),0.0],[np.sin(np.sin(gt_box[6])),-np.cos(gt_box[6]),0.0],[0,0.0,1]]),
                    gt_box[3:6],
                )
                obb.color = [1,0,0]
                vis.add_geometry(obb,reset_bounding_box=False)
            for pred_box,pred_label in zip(pred_boxes,pred_labels):
                obb = o3d.geometry.OrientedBoundingBox(
                    pred_box[0:3],
                    np.array([[np.cos(pred_box[6]),np.sin(pred_box[6]),0.0],[np.sin(np.sin(pred_box[6])),-np.cos(pred_box[6]),0.0],[0,0.0,1]]),
                    pred_box[3:6],
                )
                if pred_label == 1: # vehicle
                    obb.color = [0,1,0]
                elif pred_label == 2: #pedestrean
                    obb.color = [0.2,0.2,1]
                elif pred_label == 3: #cyclist
                    obb.color = [1,1,0]
                vis.add_geometry(obb,reset_bounding_box=False)

            current_cloud.points = o3d.utility.Vector3dVector(points)
            current_cloud.paint_uniform_color([1,1,1])
            vis.add_geometry(current_cloud,reset_bounding_box=first)
            first=False
            update =False
        #vis.update_geometry(current_cloud)
        vis.poll_events()
        vis.update_renderer()
        # 毎フレーム処理
    # dataset_len = len(test_loader)
    # # start evaluation
    # for i, batch_dict in enumerate(test_loader):
    #     load_data_to_gpu(batch_dict)
    #     with torch.no_grad():
    #         pred_dicts, ret_dict = model(batch_dict)
    




def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    logger = common_utils.create_logger(None, rank=cfg.LOCAL_RANK)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        logger=logger,
        dist=dist_test, workers=args.workers, training=False
    )
    WD = WaymoDataset(cfg.DATA_CONFIG,cfg.CLASS_NAMES,False,None,logger)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    if not os.path.exists('capture'):
        os.mkdir('capture')
    with torch.no_grad():
        visualize(model, test_loader,args, logger,WD, dist_test=dist_test)


if __name__ == '__main__':
    main()
