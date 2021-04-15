import os
import os.path as osp
import sys
import numpy as np

class Config:
 
    ## experiment
    pose_gate = True # use pose gate

    ## dataset
    dataset = 'Kinetics' # NTU, Kinetics, Mimetics
    
    ## model setting
    img_resnet_type = 18 # 18, 34, 50
    pose_resnet_type = 18 # 18, 34, 50
    resnet_feat_dim = {18: 512, 34: 512, 50: 2048}
    agg_feat_dim = 512

    ## input config
    if dataset == 'NTU':
        video_shape = (240,320)
    else:
        video_shape = (256,)
    input_img_shape = (224, 224)
    input_hm_shape = (56, 56)
    hm_sigma = 0.5
    paf_sigma = 1
    top_k_pose = 5
    frame_per_seg = 8
    frame_interval = 8
    pose_frame_factor = 1
    pose_score_thr = 0.1
        
    ## training config
    if mode == 'rgb_only' or mode == 'pose_only':
        lr = 5e-3
    else:
        lr = 5e-4
    lr_dec_epoch = [21,26]
    end_epoch = 30
    lr_dec_factor = 10
    momentum = 0.9
    weight_decay = 1e-4
    train_batch_size = 8
    train_aug = True
    reg_weight = 1.5

    ## testing config
    test_infer_num = 10
    test_batch_size = 16

    ## others
    num_thread = 40
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    
    def set_args(self, gpu_ids, mode, continue_train=False):
        self.gpu_ids = gpu_ids
        self.mode = mode
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
