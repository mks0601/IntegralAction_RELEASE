import os
import os.path as osp
from glob import glob
from tqdm import tqdm

video_root_path = './videos'
frame_save_path = './frames'
seq_path_list = glob(osp.join(video_root_path, '*'))
seq_path_list = [name for name in seq_path_list if osp.isdir(name)]

for seq_path in tqdm(seq_path_list):
    seq_name = seq_path.split('/')[-1]
    video_path_list = glob(osp.join(seq_path, '*'))
    
    for video_path in video_path_list:
        video_name = video_path.split('/')[-1]
        idx = [i for i,x in enumerate(video_name) if x == '.']
        video_name = video_name[:idx[0]]
        
        # make folders
        os.makedirs(osp.join(frame_save_path, seq_name), exist_ok=True)
        os.makedirs(osp.join(frame_save_path, seq_name, video_name), exist_ok=True)
        
        # save frames from the video
        img_path = osp.join(frame_save_path, seq_name, video_name, '%06d.jpg')
        cmd = 'ffmpeg -i ' + video_path + ' -qscale:v 1 ' + img_path
        #print(cmd)
        os.system(cmd)
