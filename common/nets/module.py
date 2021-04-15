import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_conv_layers, make_conv3d_layers, make_linear_layers

class Pose2Feat(nn.Module):
    def __init__(self, joint_num, skeleton):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.skeleton_num = len(skeleton)
        self.conv = make_conv_layers([self.joint_num+2*self.skeleton_num,64])

    def forward(self, pose_heatmap, pose_paf):
        pose_feat = torch.cat((pose_heatmap, pose_paf),1)
        pose_feat = self.conv(pose_feat)
        return pose_feat


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()
        self.img_resnet_dim = cfg.resnet_feat_dim[cfg.img_resnet_type]
        self.pose_resnet_dim = cfg.resnet_feat_dim[cfg.pose_resnet_type]

        ## temporal strided conv to fuse with RGB
        self.pose_frame_num = (cfg.frame_per_seg-1) * cfg.pose_frame_factor + 1
        self.pose_temporal_conv = make_conv3d_layers([self.pose_resnet_dim, self.pose_resnet_dim], kernel=(5,1,1), stride=(cfg.pose_frame_factor,1,1), padding=(2,0,0))

        ## pose gate layer
        self.pose_gate_fc = make_linear_layers([self.pose_resnet_dim, cfg.agg_feat_dim], relu_final=False)

        ## aggregation layer
        self.img_conv = make_conv_layers([self.img_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)
        self.img_norm = nn.LayerNorm([cfg.agg_feat_dim, 1, 1])
        self.pose_conv = make_conv_layers([self.pose_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)
        self.pose_norm = nn.LayerNorm([cfg.agg_feat_dim, 1, 1])


    def forward(self, video_feat, pose_feat):
        pose_feat = pose_feat.mean((2,3))[:,:,None,None]
        video_feat = video_feat.mean((2,3))[:,:,None,None]

        # temporal fusing with RGB
        if cfg.pose_frame_factor > 1:
            batch_size, pose_feat_dim, pose_feat_height, pose_feat_width = pose_feat.shape[0] // self.pose_frame_num, pose_feat.shape[1], pose_feat.shape[2], pose_feat.shape[3]
            pose_feat = pose_feat.view(batch_size, self.pose_frame_num, pose_feat_dim, pose_feat_height, pose_feat_width).permute(0,2,1,3,4)
            pose_feat = self.pose_temporal_conv(pose_feat)
            pose_feat = pose_feat.permute(0,2,1,3,4).reshape(-1,pose_feat_dim, pose_feat_height, pose_feat_width)
       
        # pose gate estimator
        if cfg.pose_gate:
            pose_gate = torch.sigmoid(self.pose_gate_fc(torch.squeeze(pose_feat)))[:,:,None,None]
            
            pose_feat = self.pose_conv(pose_feat)
            pose_feat = self.pose_norm(pose_feat)
            video_feat = self.img_conv(video_feat)
            video_feat = self.img_norm(video_feat)
            
            pose_feat = pose_feat * pose_gate
            video_feat = video_feat * (1 - pose_gate)
        else:
            pose_gate = None
            pose_feat = self.pose_conv(pose_feat)
            video_feat = self.img_conv(video_feat)

        # aggregation
        feat = video_feat + pose_feat
        return feat, pose_gate

class Classifier(nn.Module):
    def __init__(self, class_num):
        super(Classifier, self).__init__()
        self.class_num = class_num
        if cfg.mode == 'rgb_only':
            self.fc = make_linear_layers([cfg.resnet_feat_dim[cfg.img_resnet_type], self.class_num], relu_final=False)
        elif cfg.mode == 'pose_only':
            self.fc = make_linear_layers([cfg.resnet_feat_dim[cfg.pose_resnet_type], self.class_num], relu_final=False)
        else:
            self.fc = make_linear_layers([cfg.agg_feat_dim, self.class_num], relu_final=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, video_feat):
        video_feat = video_feat.mean((2,3))
        label_out = self.dropout(video_feat) # dropout
        label_out = self.fc(video_feat)
        return label_out

