import os
import os.path as osp
import sys
import math
import numpy as np

def clean_file(path):
    ## Clear the files under the path
    for i in os.listdir(path): 
        content_path = os.path.join(path, i) 
        if os.path.isdir(content_path):
            clean_file(content_path)
        else:
            assert os.path.isfile(content_path) is True
            os.remove(content_path)



class Config:
    # ~~~~~~~~~~~~~~~~~~~~~~Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    dataset = 'InterHand2.6M'  # InterHand2.6M  nyu hands2017
    pose_representation = '2p5D' #2p5D


    # ~~~~~~~~~~~~~~~~~~~~~~ paths~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## Please set your path
    ## Interhand2.6M dataset path. you should change to your dataset path.
    interhand_anno_dir = '/data/data1/Interhand2.6M_5fps/annotations'
    interhand_images_path = '/data/data1/Interhand2.6M_5fps/images'
    ## current file dir. change this path to your A2J-Transformer folder dir.
    cur_dir = '/data/data2/a2jformer/camera_ready'
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~input, output~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    input_img_shape = (256, 256)
    output_hm_shape = (256, 256, 256) # (depth, height, width)
    output_hm_shape_all = 256  ## For convenient
    sigma = 2.5
    bbox_3d_size = 400 # depth axis
    bbox_3d_size_root = 400 # depth axis 
    output_root_hm_shape = 64 # depth axis 


    # ~~~~~~~~~~~~~~~~~~~~~~~~backbone config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    num_feature_levels = 4
    lr_backbone = 1e-4
    masks = False
    backbone = 'resnet50' 
    dilation = True # If true, we replace stride with dilation in the last convolutional block (DC5)
    if dataset == 'InterHand2.6M':
        keypoint_num = 42
    elif dataset == 'nyu':
        keypoint_num = 14
    elif dataset == 'hands2017':
        keypoint_num = 21


    # ~~~~~~~~~~~~~~~~~~~~~~~~transformer config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    position_embedding = 'sine' #'sine' #'convLearned' # learned
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    dim_feedforward = 1024 
    enc_layers = 6
    dec_layers = 6
    pre_norm = False
    num_feature_levels = 4
    dec_n_points = 4
    enc_n_points = 4
    num_queries = 768  ## query numbers, default is 256*3 = 768 
    kernel_size = 256
    two_stage = False  ## Whether to use the two-stage deformable-detr, please select False.
    use_dab = True  ## Whether to use dab-detr, please select True.
    num_patterns = 0
    anchor_refpoints_xy = True  ##  Whether to use the anchor anchor point as the reference point coordinate, True.
    is_3D = True  # True 
    fix_anchor = True  ## Whether to fix the position of reference points to prevent update, True.
    use_lvl_weights = False  ## Whether to assign different weights to the loss of each layer, the improvement is relatively limited.
    lvl_weights = [0.1, 0.15, 0.15, 0.15, 0.15, 0.3]
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~a2j config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    RegLossFactor = 3


    # ~~~~~~~~~~~~~~~~~~~~~~~~training config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    lr_dec_epoch = [24, 35] if dataset == 'InterHand2.6M' else [45,47] 
    end_epoch = 42 if dataset == 'InterHand2.6M' else 50 
    lr = 1e-4
    lr_dec_factor = 5  
    train_batch_size = 12  
    continue_train = False  ## Whether to continue training, default is False
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~testing config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    test_batch_size = 48
    trans_test = 'gt' ## 'gt', 'rootnet' # 'rootnet' is not used


    # ~~~~~~~~~~~~~~~~~~~~~~~~dataset config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    use_single_hand_dataset = True ## Use single-handed data, default is True
    use_inter_hand_dataset = True ## Using interacting hand data, default is True
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~others~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    num_thread = 8
    gpu_ids = '0'   ## your gpu ids, for example, '0', '1-3'
    num_gpus = 1
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~directory setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    data_dir = osp.join(cur_dir, 'data')
    output_dir = osp.join(cur_dir, 'output')
    datalistDir = osp.join(cur_dir, 'datalist') ## this is used to save the dataset datalist, easy to debug.
    vis_2d_dir = osp.join(output_dir, 'vis_2d')
    vis_3d_dir = osp.join(output_dir, 'vis_3d')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    model_dir = osp.join(output_dir, 'model_dump')
    tensorboard_dir = osp.join(output_dir, 'tensorboard_log')
    clean_tensorboard_dir = False 
    clean_log_dir = False
    if clean_tensorboard_dir is True:
        clean_file(tensorboard_dir)
    if clean_log_dir is True:
        clean_file(log_dir)


    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.datalistDir)
make_folder(cfg.model_dir)
make_folder(cfg.vis_2d_dir)
make_folder(cfg.vis_3d_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
make_folder(cfg.tensorboard_dir)