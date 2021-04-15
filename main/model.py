import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.tsm.tsm_resnet import ResNetBackbone
from nets.module import Pose2Feat, Aggregator, Classifier
from nets.loss import CELoss, BCELoss
from config import cfg

class Model(nn.Module):
    def __init__(self, img_backbone, pose_backbone, pose2feat, aggregator, classifier, class_num, joint_num, skeleton):
        super(Model, self).__init__()
        self.img_backbone = img_backbone
        self.pose_backbone = pose_backbone
        self.pose2feat = pose2feat
        self.aggregator = aggregator
        self.classifier = classifier
        self.ce_loss = CELoss()
        self.bce_loss = BCELoss()

        self.class_num = class_num
        self.joint_num = joint_num
        self.skeleton_part = torch.LongTensor(skeleton).cuda().view(-1,2)
  
    def render_gaussian_heatmap(self, pose_coord, pose_score):
        x = torch.arange(cfg.input_hm_shape[1])
        y = torch.arange(cfg.input_hm_shape[0])
        yy,xx = torch.meshgrid(y,x)
        xx = xx[None,None,None,:,:].cuda().float(); yy = yy[None,None,None,:,:].cuda().float();
        
        x = pose_coord[:,:,:,0,None,None]; y = pose_coord[:,:,:,1,None,None]; 
        heatmap = torch.exp(-(((xx-x)/cfg.hm_sigma)**2)/2 -(((yy-y)/cfg.hm_sigma)**2)/2) * (pose_score[:,:,:,None,None] > cfg.pose_score_thr).float() # score thresholding
        heatmap = heatmap.sum(1) # sum overall all persons
        heatmap[heatmap > 1] = 1 # threshold up to 1
        return heatmap
    
    def render_paf(self, pose_coord, pose_score):
        x = torch.arange(cfg.input_hm_shape[1])
        y = torch.arange(cfg.input_hm_shape[0])
        yy,xx = torch.meshgrid(y,x)
        xx = xx[None,None,None,:,:].cuda().float(); yy = yy[None,None,None,:,:].cuda().float();
        
        # calculate vector between skeleton parts
        coord0 = pose_coord[:,:,self.skeleton_part[:,0],:] # batch_size*frame_num, person_num, part_num, 2
        coord1 = pose_coord[:,:,self.skeleton_part[:,1],:]
        vector = coord1 - coord0
        normalizer = torch.sqrt(torch.sum(vector**2,3,keepdim=True))
        normalizer[normalizer==0] = -1
        vector = vector / normalizer # normalize to unit vector
        vector_t = torch.stack((vector[:,:,:,1], -vector[:,:,:,0]),3)
        
        # make paf
        dist = vector[:,:,:,0,None,None] * (xx - coord0[:,:,:,0,None,None]) + vector[:,:,:,1,None,None] * (yy - coord0[:,:,:,1,None,None])
        dist_t = torch.abs(vector_t[:,:,:,0,None,None] * (xx - coord0[:,:,:,0,None,None]) + vector_t[:,:,:,1,None,None] * (yy - coord0[:,:,:,1,None,None]))
        mask1 = (dist >= 0).float(); mask2 = (dist <= normalizer[:,:,:,0,None,None]).float(); mask3 = (dist_t <= cfg.paf_sigma).float()
        score0 = pose_score[:,:,self.skeleton_part[:,0],None,None]
        score1 = pose_score[:,:,self.skeleton_part[:,1],None,None]
        mask4 = ((score0 >= cfg.pose_score_thr) * (score1 >= cfg.pose_score_thr)) # socre thresholding
        mask = mask1 * mask2 * mask3 * mask4 # batch_size*frame_num, person_num, part_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        paf = torch.stack((mask * vector[:,:,:,0,None,None], mask * vector[:,:,:,1,None,None]),3) # batch_size*frame_num, person_num, part_num, 2, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        
        # sum and normalize
        mask = torch.sum(mask, (1))
        mask[mask==0] = -1
        paf = torch.sum(paf, (1)) / mask[:,:,None,:,:] # batch_size*frame_num, part_num, 2, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        paf = paf.view(paf.shape[0], paf.shape[1]*paf.shape[2], paf.shape[3], paf.shape[4])
        return paf 

    def forward(self, inputs, targets, meta_info, mode):
        input_video = inputs['video'] # batch_size, frame_num, 3, cfg.input_img_shape[0], cfg.input_img_shape[1]
        batch_size, video_frame_num = input_video.shape[:2]
        input_video = input_video.view(batch_size*video_frame_num, 3, cfg.input_img_shape[0], cfg.input_img_shape[1])

        pose_coords = inputs['pose_coords'] # batch_size, frame_num, person_num, joint_num, 2
        pose_scores = inputs['pose_scores'] # batch_size, frame_num, person_num, joint_num
        batch_size, pose_frame_num = pose_coords.shape[:2]
        pose_coords = pose_coords.view(batch_size*pose_frame_num, cfg.top_k_pose, self.joint_num, 2)
        pose_scores = pose_scores.view(batch_size*pose_frame_num, cfg.top_k_pose, self.joint_num)
        input_pose_hm = self.render_gaussian_heatmap(pose_coords, pose_scores) # batch_size*pose_frame_num, self.joint_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        input_pose_paf = self.render_paf(pose_coords, pose_scores) # batch_size*pose_frame_num, 2*part_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]

        # rgb only
        if cfg.mode == 'rgb_only':
            video_feat = self.img_backbone(input_video, skip_early=False)
            action_label_out = self.classifier(video_feat)
            action_label_out = action_label_out.view(batch_size, video_frame_num, -1).mean(1)
        # pose only
        elif cfg.mode == 'pose_only':
            pose_feat = self.pose2feat(input_pose_hm, input_pose_paf)
            pose_feat = self.pose_backbone(pose_feat, skip_early=True)
            action_label_out = self.classifier(pose_feat)
            action_label_out = action_label_out.view(batch_size, pose_frame_num, -1).mean(1)
        # pose late fusion
        elif cfg.mode == 'rgb+pose':
            video_feat = self.img_backbone(input_video, skip_early=False)
            pose_feat = self.pose2feat(input_pose_hm, input_pose_paf)
            pose_feat = self.pose_backbone(pose_feat, skip_early=True)
            video_pose_feat, pose_gate = self.aggregator(video_feat, pose_feat)
            action_label_out = self.classifier(video_pose_feat)
            action_label_out = action_label_out.view(batch_size, video_frame_num, -1).mean(1)
            
        if mode == 'train':
            # loss functions
            loss = {}
            loss['action_cls'] = self.ce_loss(action_label_out, targets['action_label'])
            if cfg.mode == 'rgb+pose' and cfg.pose_gate: loss['pose_gate'] = self.bce_loss(pose_gate, torch.ones_like(pose_gate)) * cfg.reg_weight
            return loss

        else:
            # test output
            out = {}
            out['action_prob'] = F.softmax(action_label_out,1)
            out['img_id'] = meta_info['img_id']
            if cfg.mode == 'rgb+pose' and cfg.pose_gate: out['pose_gate'] = pose_gate.view(batch_size, -1).mean((1))
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(class_num, joint_num, skeleton, mode):
    img_backbone = ResNetBackbone(cfg.img_resnet_type, cfg.frame_per_seg)
    pose_backbone = ResNetBackbone(cfg.pose_resnet_type, (cfg.frame_per_seg-1)*cfg.pose_frame_factor+1)
    pose2feat = Pose2Feat(joint_num, skeleton)
    aggregator = Aggregator()
    classifier = Classifier(class_num)

    if mode == 'train':
        img_backbone.init_weights()
        pose_backbone.init_weights()
        pose2feat.apply(init_weights)
        aggregator.apply(init_weights)
        classifier.apply(init_weights)
   
    model = Model(img_backbone, pose_backbone, pose2feat, aggregator, classifier, class_num, joint_num, skeleton)
    return model

