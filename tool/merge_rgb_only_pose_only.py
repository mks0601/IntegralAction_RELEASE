import torch
import os
import os.path as osp

# load rgb_only model
rgb_only_path = osp.join('.', 'snapshot_29_rgb_only.pth.tar')
rgb_only = torch.load(rgb_only_path)

# remove pose modules, agg, and cls modules from rgb_only model
dump_key_rgb = []
for k,v in rgb_only['network'].items():
    if 'aggregator' in k or 'classifier' in k or 'pose_backbone' in k or 'pose2feat' in k:
        print('rgb_only', k)
        dump_key_rgb.append(k)
for k in dump_key_rgb:
    rgb_only['network'].pop(k, None)
rgb_only['epoch'] = 0

# load pose_only model
pose_only_path = osp.join('.', 'snapshot_29_pose_only.pth.tar')
pose_only = torch.load(pose_only_path)

# remove rgb modules, agg, and cls modules frompose_only model
dump_key_pose = []
for k,v in pose_only['network'].items():
    if 'aggregator' in k or 'classifier' in k or 'img_backbone' in k:
        print('pose_only', k, v.shape)
        dump_key_pose.append(k)
for k in dump_key_pose:
    pose_only['network'].pop(k, None)
pose_only['epoch'] = 0

# merge rgb_only and pose_only
rgb_only['network'] = {**rgb_only['network'], **pose_only['network']}

save_path = osp.join('.', 'snapshot_0.pth.tar')
torch.save(rgb_only, save_path)
