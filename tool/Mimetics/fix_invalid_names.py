import os
import os.path as osp
from glob import glob
from tqdm import tqdm

video_root_path = './videos'
seq_path_list = glob(osp.join(video_root_path, '*'))
seq_path_list = [name for name in seq_path_list if osp.isdir(name)]
blacklist = [' ', '(', ')', "'", '"']

for seq_path in tqdm(seq_path_list):
    seq_name = seq_path.split('/')[-1]
    if len([x for x in seq_name if x in blacklist]) == 0:
        continue
    
    seq_name_invalid = ''.join([x if x not in blacklist else '\\' + x for x in seq_name])
    seq_name_valid = ''.join([x if x not in blacklist else '_' for x in seq_name])

    cmd = 'mv ' + osp.join(video_root_path, seq_name_invalid) + ' ' + osp.join(video_root_path, seq_name_valid)
    #print(cmd)
    os.system(cmd)
