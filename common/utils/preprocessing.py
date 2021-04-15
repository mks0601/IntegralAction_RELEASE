import numpy as np
import cv2
import random
from config import cfg
import math
import os.path as osp

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def load_video(path, frame_num, start_frame_idx):
    
    if start_frame_idx is None: # random sample
        start_frame_idx = random.randrange(0, frame_num - (cfg.frame_per_seg-1) * cfg.frame_interval) # 0-based
    
    # load frames
    video = []; video_frame_idxs = [];     
    for i in range(cfg.frame_per_seg):
        cur_frame_idx = start_frame_idx + i * cfg.frame_interval # 0-based
        img = load_img(osp.join(path, '%.6d.jpg' % (cur_frame_idx+1))) # 1-based
        if len(cfg.video_shape) == 2:
            img = cv2.resize(img, (cfg.video_shape[1], cfg.video_shape[0]))
            video_shape = cfg.video_shape
        else: # resize shorter side of video to cfg.video_shape
            height, width = img.shape[:2]
            if width > height:
                video_shape = (cfg.video_shape[0], int(width / height * cfg.video_shape[0])) # height, width
            else:
                video_shape = (int(height / width * cfg.video_shape[0]), cfg.video_shape[0]) # height, width
            img = cv2.resize(img, (video_shape[1], video_shape[0]))
        video.append(img)
        video_frame_idxs.append(cur_frame_idx)
    video = np.array(video).reshape(cfg.frame_per_seg, video_shape[0], video_shape[1], 3)

    # frame indexs for pose sampling
    pose_frame_idxs = [i for i in range(start_frame_idx, start_frame_idx + (cfg.frame_per_seg-1) * cfg.frame_interval + 1, cfg.frame_interval//cfg.pose_frame_factor)]
    return video, pose_frame_idxs

def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, img_width, img_height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

def process_skeleton(pose_coords, pose_scores, img2aug_trans, do_flip, flip_pairs, joint_num, resized_shape):
    frame_num, person_num = pose_coords.shape[:2]
    pose_coords = pose_coords.reshape(-1,2)
    pose_scores = pose_scores.reshape(-1)

    # apply affine flip and affine transformation
    if do_flip:
        pose_coords[:,0] = resized_shape[1] - pose_coords[:,0] - 1
        for pair in flip_pairs:
            pose_coords[pair[0], :], pose_coords[pair[1], :] = pose_coords[pair[1], :], pose_coords[pair[0], :].copy()
            pose_scores[pair[0]], pose_scores[pair[1]] = pose_scores[pair[1]], pose_scores[pair[0]].copy()
    pose_coords_xy1 = np.concatenate((pose_coords, np.ones_like(pose_coords[:,0:1])),1)
    pose_coords = np.dot(img2aug_trans, pose_coords_xy1.transpose(1,0)).transpose(1,0)[:,:2]

    # transform to input heatmap space
    pose_coords[:,0] = pose_coords[:,0] / cfg.input_img_shape[1] * cfg.input_hm_shape[1]
    pose_coords[:,1] = pose_coords[:,1] / cfg.input_img_shape[0] * cfg.input_hm_shape[0]

    pose_coords = pose_coords.reshape(frame_num, person_num, joint_num, 2)
    pose_scores = pose_scores.reshape(frame_num, person_num, joint_num)
    
    return pose_coords, pose_scores

def get_aug_config(video_shape):
    # ready for random scale cropping
    scale_factor = [1, .875, .75, .66]
    scale = []
    for i,w in enumerate(scale_factor):
        for j,h in enumerate(scale_factor):
            if abs(i-j) <= 1:
                scale.append((w, h))
    scale = random.choice(scale)
    width, height = int(scale[0] * cfg.input_img_shape[1]), int(scale[1] * cfg.input_img_shape[0])
    
    # make bbox
    xmin = random.randint(0, video_shape[1] - width)
    ymin = random.randint(0, video_shape[0] - height)
    bbox = [xmin, ymin, width, height] # xmin, ymin, width, height

    # horizontal flip augmentation
    do_flip = random.random() <= 0.5
    return bbox, do_flip

def augmentation(video, data_split):
    video_shape = video.shape[1:3] # height, width

    if data_split == 'train' and cfg.train_aug:
        bbox, do_flip = get_aug_config(video_shape)
    else:
        xmin = video_shape[1]//2 - cfg.input_img_shape[1]//2
        ymin = video_shape[0]//2 - cfg.input_img_shape[0]//2
        bbox = [xmin, ymin, cfg.input_img_shape[1], cfg.input_img_shape[0]] # xmin, ymin, width, height
        do_flip = False
    
    frame_num = video.shape[0]
    video_aug = np.zeros((frame_num, cfg.input_img_shape[0], cfg.input_img_shape[1], 3), dtype=np.float32)
    for i in range(frame_num):
        video_aug_before_resize, trans, inv_trans = generate_patch_image(video[i], bbox, do_flip, cfg.input_img_shape)
        video_aug[i] = cv2.resize(video_aug_before_resize, (cfg.input_img_shape[1], cfg.input_img_shape[0]))
    video = video_aug

    return video, trans, inv_trans, do_flip

def generate_patch_image(cvimg, bbox, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0])
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], inv=True)

    return img_patch, trans, inv_trans

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, inv=False):
    src_w = src_width
    src_h = src_height
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

