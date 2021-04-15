import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.preprocessing import load_video, augmentation, process_skeleton
from utils.vis import vis_keypoints

class NTU(torch.utils.data.Dataset):
    def __init__(self, data_split):
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'NTU', 'data')
        self.class_num = 60
        self.joint_num = 25
        self.joint_names = ('Pelvis', 'Chest', 'Neck', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand1', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand1', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'Thorax', 'R_Hand2', 'R_Hand3', 'L_Hand2', 'L_Hand3')
        self.flip_pairs = ( (4,8), (5,9), (6,10), (7,11), (12,16), (13,17), (14,18), (15,19), (21,23), (22,24) )
        self.skeleton = ( (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20) )
        self.datalist = self.load_data()
        
    def load_data(self):
        db = COCO(osp.join(self.data_path, 'NTU_' + self.data_split + '.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            setup_idx = img['setup_idx']
            img_path = osp.join(self.data_path, 'frames', 'nturgb+d_rgb_s' + str(setup_idx), img['file_name'])
            skeleton_path = osp.join(self.data_path, 'nturgb+d_skeletons', img['skeleton_file_name'])
            img_height = img['height']; img_width = img['width'];
            frame_num = img['frame_num']
            action_label = ann['action_label']

            # exclude video with small number of frames
            if frame_num <= (cfg.frame_per_seg-1) * cfg.frame_interval:
                continue
            
            if self.data_split == 'train':
                data_dict = {'img_id': image_id, 'img_path': img_path, 'skeleton_path': skeleton_path, 'original_shape': (img_height, img_width), 'action_label': action_label, 'frame_num': frame_num, 'start_frame_idx': None}
                datalist.append(data_dict)
            else:
                assert cfg.test_infer_num > 1, print('cfg.test_infer_num should be larger than 1')
                for t in range(cfg.test_infer_num):
                    start_frame_idx = int((frame_num - 1 - (cfg.frame_per_seg-1) * cfg.frame_interval) * t / (cfg.test_infer_num - 1))
                    data_dict = {'img_id': image_id, 'img_path': img_path, 'skeleton_path': skeleton_path, 'original_shape': (img_height, img_width), 'action_label': action_label, 'frame_num': frame_num, 'start_frame_idx': start_frame_idx}
                    datalist.append(data_dict)
        
        return datalist
    
    def load_skeleton(self, skeleton_path, frame_idxs, original_shape, resized_shape):
        with open(skeleton_path) as f:
            skeleton_data = f.readlines()

        line_idx = 0
        cur_fid = 0
        pose_coords = np.ones((len(frame_idxs), cfg.top_k_pose, self.joint_num, 2), dtype=np.float32) # joint coordinates from all frames in frame_idxs
        pose_scores = np.zeros((len(frame_idxs), cfg.top_k_pose, self.joint_num), dtype=np.float32)
        while cur_fid <= frame_idxs[-1]:
            line_idx += 1
            person_num = int(skeleton_data[line_idx])

            if cur_fid in frame_idxs:
                pose_coords_per_frame = np.ones((person_num, self.joint_num, 2), dtype=np.float32) # joint coordinates per frame
                pose_scores_per_frame = np.zeros((person_num, self.joint_num), dtype=np.float32)
                for pid in range(person_num):
                    line_idx += 2
                    pose_coord = np.ones((self.joint_num,2), dtype=np.float32)
                    pose_score = np.ones((self.joint_num), dtype=np.float32)
                    for j in range(self.joint_num):
                        line_idx += 1
                        pose_coord[j][0] = float(skeleton_data[line_idx].split()[5])
                        pose_coord[j][1] = float(skeleton_data[line_idx].split()[6])
                        
                        # There are some NaN joint coordinates in skeleton annotation files
                        if math.isnan(pose_coord[j][0]) or math.isnan(pose_coord[j][1]):
                            pose_coord[j] = 0
                            pose_score[j] = 0

                    # resize to video_shape
                    pose_coord[:,0] = pose_coord[:,0] / original_shape[1] * resized_shape[1]
                    pose_coord[:,1] = pose_coord[:,1] / original_shape[0] * resized_shape[0]

                    pose_coords_per_frame[pid] = pose_coord
                    pose_scores_per_frame[pid] = pose_score
                
                # select top-k pose
                if person_num < cfg.top_k_pose:
                    pose_coords_per_frame = np.concatenate((pose_coords_per_frame, np.ones((cfg.top_k_pose - person_num, self.joint_num, 2))))
                    pose_scores_per_frame = np.concatenate((pose_scores_per_frame, np.zeros((cfg.top_k_pose - person_num, self.joint_num))))
                top_k_idx = np.argsort(np.mean(pose_scores_per_frame,1))[-cfg.top_k_pose:][::-1]
                pose_coords_per_frame = pose_coords_per_frame[top_k_idx,:,:]
                pose_scores_per_frame = pose_scores_per_frame[top_k_idx,:]
                
                idx = frame_idxs.index(cur_fid)
                pose_coords[idx] = pose_coords_per_frame # save pose_coord of cur_fid frame
                pose_scores[idx] = pose_scores_per_frame
                cur_fid += 1
            else:
                line_idx += person_num * (2 + self.joint_num )
                cur_fid += 1
        
        return pose_coords, pose_scores
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, skeleton_path, original_shape, action_label, frame_num, start_frame_idx = data['img_path'], data['skeleton_path'], data['original_shape'], data['action_label'], data['frame_num'], data['start_frame_idx']
        
        # video load
        video, skeleton_frame_idxs = load_video(img_path, frame_num, start_frame_idx)
        resized_shape = video.shape[1:3]

        # augmentation
        video, img2aug_trans, aug2img_trans, do_flip = augmentation(video, self.data_split)
        video = video.transpose(0,3,1,2).astype(np.float32)/255. # frame_num, channel_dim, height, width

        # skeleton information load
        pose_coords, pose_scores = self.load_skeleton(skeleton_path, skeleton_frame_idxs, original_shape, resized_shape)
        
        # process skeleton information
        pose_coords, pose_scores = process_skeleton(pose_coords, pose_scores, img2aug_trans, do_flip, self.flip_pairs, self.joint_num, resized_shape)
        
        """
        # for debug
        # keypoint visualization
        for i in range(cfg.frame_per_seg):
            img = video[i,::-1,:,:].transpose(1,2,0) * 255
            person_num = len(pose_coords[i])
            for p in range(person_num):
                #for j in range(self.joint_num):
                    #coord = (int(pose_coords[i][p][j][0]), int(pose_coords[i][p][j][1]))
                    #cv2.circle(img, coord, radius=3, color=(255,0,0), thickness=-1, lineType=cv2.LINE_AA)
                    #cv2.imwrite(str(idx) + '_' + str(action_label) + '_' + str(i) + '_' + str(j) + '.jpg', img)
                coord = pose_coords[i][p].copy()
                coord[:,0] = coord[:,0] / cfg.input_hm_shape[1] * cfg.input_img_shape[1]
                coord[:,1] = coord[:,1] / cfg.input_hm_shape[0] * cfg.input_img_shape[0]
                img = vis_keypoints(img, pose_coords[i][p] * 4, self.skeleton)
            cv2.imwrite(str(idx) + '_' + str(action_label) + '_' + str(i) + '.jpg', img)
        """
        
        inputs = {'video': video, 'pose_coords': pose_coords, 'pose_scores': pose_scores}
        targets = {'action_label': action_label}
        meta_info = {'img_id': data['img_id']}
        return inputs, targets, meta_info

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)
        
        action_label_out = {}
        action_label_gt = {}
        for n in range(sample_num):
            annot = annots[n]
            out = outs[n]

            assert int(annot['img_id']) == int(out['img_id'])
            img_id = annot['img_id']

            # gt
            if img_id not in action_label_gt:
                action_label_gt[img_id] = annot['action_label']

            # output
            if img_id not in action_label_out:
                action_label_out[img_id] = [out['action_prob']]
            else:
                action_label_out[img_id].append(out['action_prob'])

        # average multiple inferences on a single video
        for k in action_label_out.keys():
            action_label_out[k] = sum(action_label_out[k]) / len(action_label_out[k])
        
        # measure accuracy
        sample_num = len(action_label_out)
        top_1 = 0; top_5 = 0;
        for k in action_label_gt.keys():
            if action_label_gt[k] == np.argmax(action_label_out[k]):
                top_1 += 1
            if action_label_gt[k] in np.argsort(action_label_out[k])[-5:][::-1]:
                top_5 += 1

        print('Top-1 accuracy: %.4f' % (top_1 / sample_num * 100))
        print('Top-5 accuracy: %.4f' % (top_5 / sample_num * 100))

        # result save
        for k in action_label_out.keys():
            action_label_out[k] = action_label_out[k].tolist()
        output_path = osp.join(cfg.result_dir, 'ntu_result.json')
        with open(output_path, 'w') as f:
            json.dump(action_label_out, f)
        print('Result is saved at: ' + output_path)
        

