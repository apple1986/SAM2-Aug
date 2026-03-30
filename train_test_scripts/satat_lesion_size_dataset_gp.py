import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
import numpy as np
from os.path import join
from os import makedirs, listdir
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import SimpleITK as sitk
import os
import argparse
import shutil

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

from torch import multiprocessing as mp
from torchvision import transforms
from dataset.ustwlits import ToTensor, CenterCrop
from builders.utswdataset_builder import load_data
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand, get_all_bounding_boxes_with_ccl
from gp_knife.freq_func import set_random_seed
from gp_knife.eval_model import get_dice_hd_nsd_asd
from medpy.metric import dc
from scipy.ndimage import label
## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
###########################
## set parameters
parser = argparse.ArgumentParser(
    description="Run inference on validation set with MedSAM2"
)
parser.add_argument(
    "-data_root",
    type=str,
    default=None,
    help="Path to the data folder",
)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/check_results",
    help="Path to save the segmentation results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default='/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt',
    help="SAM2 pretrained model checkpoint",
)
parser.add_argument(
    "-model_cfg",
    type=str,
    default="sam2_hiera_t.yaml",
    help="Model config file"
)
parser.add_argument("-device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-bbox_shift",
    type=int,
    default=5,
    help="Bounding box perturbation",
)
parser.add_argument(
    "-num_workers",
    type=int,
    default=16,
    help="Number of workers for multiprocessing",
)
parser.add_argument("--visualize", default=True, help="Save the .nii.gz slice segmentation results")
args = parser.parse_args()


def analyze_3d_mask(mask):
    """
    Analyzes a 3D binary mask and returns the number of connected objects
    and the number of voxels in each object.
    """
    # Step 2: Label the connected components in the 3D volume
    labeled_mask, num_objects = label(mask)
    print(f"Number of connected objects: {num_objects}")

    # Step 3 and 4: Calculate the voxel count for each object
    object_sizes = [(i, np.sum(labeled_mask == i)) for i in range(1, num_objects + 1)]
    
    print("Object ID - Voxel Count")
    for obj_id, size in object_sizes:
        print(f"Object {obj_id}: {size} voxels")
        
    return labeled_mask, object_sizes

#######################################
## testing
image_size = 1024
args.visualize = False
data_root = args.data_root
# bbox_shift = args.bbox_shift
args.bbox_shift = 5
args.noMask = False
device = "cuda:0"  # args.device
# args.model_cfg = "sam2_hiera_s.yaml"
# args.sam2_checkpoint = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_small.pt"
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
args.ch3 = True
flag_fix = True # False True
tv_types = ['itv',] # 'gtv',  'ptv', 'alltv'
# tv_types = ['itv']

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

for tv_type in tv_types:
    for NUM in range(5, 6):
        model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir/paper_used_model/Sam2_Tiny_Promp_Aug_3Ch/AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand"+str(NUM)
        # medsam2_checkpoint = os.path.join(model_path, "medsam_model_best_dice.pth")
        num_workers = args.num_workers
        root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2"
        save_eval_path = os.path.join(model_path, "eval_result_all_"+tv_type)
        makedirs(save_eval_path, exist_ok=True)
        # save segmentation results
        # pred_save_dir = os.path.join(model_path, "seg_result_"+tv_type)
        # makedirs(pred_save_dir, exist_ok=True)
        ## copy the current testing script
        script_path = join(model_path, os.path.basename(__file__))
        if not os.path.exists(script_path):
            shutil.copyfile(
                __file__, script_path
            )
        args.bbox_shift =  NUM # NUM
        
        sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)

        ## load data
        all_test_transform = transforms.Compose([
                # CenterCrop(patch_size),
                ToTensor(),
        ])
        testLoader = load_data(root_path, all_test_transform, batch_size=1, split="test", tv_type=tv_type)

        dice_all = [] # save all data
        hd95_all = []
        nsd_all = []
        asd_all = []
        cnt_case = 0
        lesion_vox_num = []
        for data_four in testLoader:
            cnt_case = cnt_case + 1
            print(f"cnt case {cnt_case}")
            _, _, _, curGT = data_four["curMR"], data_four["preMR"], data_four["preGT"], data_four["curGT"]
            caseID = data_four["caseID"][0]

            curGT_np = torch.permute(curGT, dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
            labeled_mask, object_sizes = analyze_3d_mask(curGT_np)
            for n in range(len(object_sizes)):
                lesion_vox_num.append(object_sizes[n][1])
        lesion_vox_num = np.array(lesion_vox_num)

        ## save each case results to txt
        with open(os.path.join(save_eval_path, "stat_lesion_vox_mean_std_"+tv_type+".txt"), "w") as f:
            write_content = f"number of lesion: {len(lesion_vox_num)}\nmean and std\n{lesion_vox_num.mean():.0f}\t{lesion_vox_num.std():.0f}\n"
            write_content = write_content + f"{lesion_vox_num.mean():.0f}±{lesion_vox_num.std():.0f}\n"
            f.write(write_content)