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
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
args.ch3 = True
flag_fix = True # False True
# tv_types = ['brain_all', 'brain_gtv', 'brain_ctv', 'brain_ptv']
tv_types = ['cns_v2_ctv']
# tv_types = ['itv']
pre_caseID = "casexxx" # "case29" "case30" "case31" "case32" "case33"

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

for tv_type in tv_types:
    for NUM in range(5, 6):
        data_path = "/home/gxu/proj1/lesionSeg/dataset/dataset_brain_cns_v2"
        root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2"
        save_eval_path = os.path.join(data_path, "stats_eval_result_all_mri_"+tv_type)
        makedirs(save_eval_path, exist_ok=True)
        ## copy the current testing script
        script_path = join(data_path, os.path.basename(__file__))
        if not os.path.exists(script_path):
            shutil.copyfile(
                __file__, script_path
            )
        
        sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)

        ## load data
        all_test_transform = transforms.Compose([
                # CenterCrop(patch_size),
                ToTensor(),
        ])
        testLoader = load_data(root_path, all_test_transform, batch_size=1, split=tv_type, tv_type=tv_type)

        dice_all = [] # save all data
        hd95_all = []
        nsd_all = []
        asd_all = []
        cnt_case = 0
        lesion_vox_num = []
        lesion_res_vox_num = []
        vol_shape = []
        for data_four in testLoader:
            cnt_case = cnt_case + 1
            print(f"cnt case {cnt_case}")
            _, _, preGT, curGT = data_four["curMR"], data_four["preMR"], data_four["preGT"], data_four["curGT"]
            cur_caseID = data_four["caseID"][0].split("_")[0] # get the caseID, e.g., case29
            resolution_scale = data_four["volume_vox_res"][0].item() # get the resolution scale, e.g., 3.6 mm^3
            print(f"current caseID: {cur_caseID}, resolution_scale: {resolution_scale}")
            vol_shape_each = curGT.shape[1:] # get the volume shape, e.g., (128, 128, 128)
            vol_shape_each = [int(i) for i in vol_shape_each] # convert to int
            vol_shape.append(vol_shape_each) # save the volume shape for each case

            # ## if the caseID is the first time, we need to calculate the first MRI
            # if cur_caseID != pre_caseID:
            #     preGT = preGT[0,...] # BCDHW -> CDHW
            #     preGT_np = torch.permute(preGT, dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
            #     labeled_mask, object_sizes = analyze_3d_mask(preGT_np)
            #     for n in range(len(object_sizes)):
            #         lesion_vox_num.append(object_sizes[n][1])
            #         lesion_res_vox_num.append(object_sizes[n][1] * resolution_scale) # resolution_scale is the scale of the image, e.g., 3.6 mm^3
            #     pre_caseID = cur_caseID

            # calculate the subsequent MRI
            curGT_np = torch.permute(curGT, dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
            labeled_mask, object_sizes = analyze_3d_mask(curGT_np)
            for n in range(len(object_sizes)):
                lesion_vox_num.append(object_sizes[n][1])
                lesion_res_vox_num.append(object_sizes[n][1] * resolution_scale) 
           
        lesion_vox_num = np.array(lesion_vox_num)
        lesion_res_vox_num = np.array(lesion_res_vox_num)
        vol_mean_shape = np.mean(vol_shape, axis=0)
        print(f"mean volume shape: {vol_mean_shape}")
        print(f"number of lesion: {len(lesion_vox_num)}")
        print(f"mean and std: {lesion_vox_num.mean():.0f}\t{lesion_vox_num.std():.0f}")
        print(f"{lesion_vox_num.mean():.0f}±{lesion_vox_num.std():.0f}")
        print(f"number of lesion with resolution: {len(lesion_res_vox_num)}")
        print(f"mean and std: {lesion_res_vox_num.mean():.0f}\t{lesion_res_vox_num.std():.0f}")
        ## save each case results to txt
        with open(os.path.join(save_eval_path, "stat_lesion_res_vox_mean_std_all_mri_"+tv_type+".txt"), "w") as f:
            write_content = f"number of lesion: {len(lesion_vox_num)}\nmean and std\n{lesion_vox_num.mean():.0f}\t{lesion_vox_num.std():.0f}\n"
            write_content = write_content + f"{lesion_vox_num.mean():.0f}±{lesion_vox_num.std():.0f}\n"
            write_content = write_content + f"number of lesion with resolution: {len(lesion_res_vox_num)}\nmean and std\n{lesion_res_vox_num.mean():.0f}\t{lesion_res_vox_num.std():.0f}\n"
            write_content = write_content + f"{lesion_res_vox_num.mean():.0f}±{lesion_res_vox_num.std():.0f}\n"
            write_content = write_content + f"mean volume shape: {vol_mean_shape}\n"
            f.write(write_content)
    
