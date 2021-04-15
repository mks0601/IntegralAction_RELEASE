import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--mode', type=str, dest='mode')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    assert args.mode, 'please enter mode from one of [rgb_only, pose_only, rgb+pose].'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, args.mode)
    cudnn.benchmark = True

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    outs = []
    with torch.no_grad():
        for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
                        
            # forward
            out = tester.model(inputs, targets, meta_info, 'test')
            
            # save output
            out = {k: v.cpu().numpy() for k,v in out.items()}
            for k,v in out.items(): batch_size = out[k].shape[0]
            out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]
            outs += out

    # evaluate
    tester._evaluate(outs)

if __name__ == "__main__":
    main()
