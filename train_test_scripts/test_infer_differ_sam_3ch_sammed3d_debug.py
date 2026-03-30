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
from radioa.utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2

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
parser.add_argument(
    "-sam_root_path",
    type=str,
    default='/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam_based',
    help="sam-based path",
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

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

#######################################
"""different sam-based models"""
from pathlib import Path
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.nn.functional as F
from typing import TypeVar
import torchio as tio
from itertools import product
import nibabel as nib


from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep
from radioa.utils.SAMMed3D_segment_anything.build_sam3D import build_sam3D_vit_b_ori
from radioa.utils.SAMMed3D_segment_anything.modeling.sam3D import Sam3D
from radioa.utils.image import get_crop_pad_params_from_gt_or_prompt
from radioa.utils.resample import get_current_spacing_from_affine, resample_to_shape, resample_to_spacing
from radioa.utils.transforms import resample_to_spacing_sparse
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib

import radioa.prompts.prompter as prmp

#####################################
def load_sammed3d(checkpoint_path, device="cuda"):
    sam_model_tune = build_sam3D_vit_b_ori(checkpoint=None)
    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = model_dict["model_state_dict"]
        sam_model_tune.load_state_dict(state_dict)
        sam_model_tune.to(device)
        sam_model_tune.eval()

    return sam_model_tune

def finetune_model_predict3D(img3D, gt3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10, prev_masks=None):
    # norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    # img3D = norm_transform(img3D.squeeze(dim=1))  # (N, C, W, H, D)
    # img3D = img3D.unsqueeze(dim=1)

    batch_size, ch, width, height, depth = img3D.shape
    pred_batch = torch.zeros((batch_size, ch, width, height, depth), device=device)

    for b in range(batch_size):
        click_points = []
        click_labels = []

        img3D_single = img3D[b:b+1,...]
        gt3D_single = gt3D[b:b+1,...]

        crop_size = 128

        if prev_masks is None:
            prev_masks = torch.zeros_like(img3D_single).to(device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size // 4, crop_size // 4, crop_size // 4))

        with torch.no_grad():
            image_embedding = sam_model_tune.image_encoder(img3D_single.to(device))

        for num_click in range(num_clicks):
            with torch.no_grad():
                if num_click > 1:
                    click_method = get_next_click3D_torch_2
                batch_points, batch_labels = click_method(prev_masks.to(device), gt3D_single.to(device))

                points_co = torch.cat(batch_points, dim=0).to(device)
                points_la = torch.cat(batch_labels, dim=0).to(device)

                click_points.append(points_co)
                click_labels.append(points_la)

                points_input = points_co
                labels_input = points_la

                sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                    points=[points_input, labels_input],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
                low_res_masks, _ = sam_model_tune.mask_decoder(
                    image_embeddings=image_embedding.to(device),
                    image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                prev_masks = F.interpolate(low_res_masks, size=img3D_single.shape[-3:], mode='trilinear', align_corners=False)
                medsam_seg_prob = prev_masks  # (B, 1, W, H, D)
                medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)

        pred_batch[b:b+1,:,:,:,:] = medsam_seg

    return pred_batch
#######################################
## set path
model_names = ["sam_med3d.pth",]
checkpoint_path = os.path.join(args.sam_root_path, model_names[0])
affine_matrix = np.array([[-0.88999999,  0.,          0.,        -0.,        ],
                            [ 0.,         -0.88999999,  0.,         -0.,        ],
                            [ 0. ,         0. ,         3.  ,        0.   ,     ],
                            [ 0. ,         0. ,         0.  ,        1.   ,     ]])
## testing
image_size = 1024
args.visualize = True
data_root = args.data_root
# bbox_shift = args.bbox_shift
args.bbox_shift = 5
args.noMask = True
device = "cuda:0"  # args.device
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
args.ch3 = True
# MedSAM2-UTSW-GP-Prompt-Mask-20250310-2015     MedSAM2-UTSW-GP-Prompt-Mask-20250310-1937   MedSAM2-UTSW-GP-FT-All-20250311-0112
flag_fix = False # False True

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

for NUM in range(5, 6):
    model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir/Sam2_Tiny_Promp_Aug_3Ch/FT_Img_Msk_Prmp_3ch_Rand"+str(NUM)
    num_workers = args.num_workers
    root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2"
    save_eval_path = os.path.join(model_path, "dataset_gp/sammed3d_b_eval_result_itv")
    makedirs(save_eval_path, exist_ok=True)
    # save segmentation results
    pred_save_dir = os.path.join(model_path, "dataset_gp/sammed3d_b_seg_result_itv")
    makedirs(pred_save_dir, exist_ok=True)

    # load fine-tuned model
    sam_model_tune = load_sammed3d(checkpoint_path=checkpoint_path, device=device)
    ## load data
    all_test_transform = transforms.Compose([
            # CenterCrop(patch_size),
            ToTensor(),
    ])
    testLoader = load_data(root_path, all_test_transform, batch_size=1, split="test")

    # load evaluation function
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    cnt_case = 0
    num_clicks = 5
    prev_masks = None
    for data_four in testLoader:
        cnt_case = cnt_case + 1
        print(f"cnt case {cnt_case}")
        curMR, preMR, preGT, curGT = data_four["curMR"], data_four["preMR"], data_four["preGT"], data_four["curGT"]
        caseID = data_four["caseID"][0]
        curMR = curMR.to(device) # BCHWD
        preMR = preMR.to(device) # BCHWD
        curGT = curGT.to(device).unsqueeze(1).float() # BHWD
        preGT = preGT.to(device) # B1HWD
        # print(f"curGT shape: {curGT.shape}")
        # print(f"curMR max value: {curMR.max()}")

        B, C, H, W, D = curMR.shape
        # curMR = torch.permute(curMR, (0,1,4,2,3)) # BCDWH
        curGT_np = curGT.cpu().numpy()[0,0, ...] # HWD
        # curMR = torch.permute(curMR, dims=(0, 1, 3, 2, 4)) # B1WHD
        # curGT = torch.permute(curGT.unsqueeze(1).float(), dims=(0, 1, 3, 2, 4)) # B3WHD

        ## inference
        batch_size, ch, width, height, depth = curMR.shape
        pred_batch = torch.zeros((batch_size, ch, width, height, depth), device=device)
        curMR = F.interpolate(curMR, size=(128, 128, 128), mode="trilinear", align_corners=False)
        curMR = (curMR * 255.0 - 123.67) / 58.39 ## ap
        curGT = F.interpolate(curGT, size=(128, 128, 128), mode="nearest", align_corners=None)
        tmp_pd = finetune_model_predict3D(
                curMR, curGT, sam_model_tune, device=device,
                click_method= get_next_click3D_torch_ritm, num_clicks=10, 
                prev_masks=None) 

        # cal dice, hd95, nsd and asd
        # pd = torch.from_numpy(tmp_pd)[None, None, ...] #BCDHW
        pd = F.interpolate(tmp_pd.float(), size=(512, 512, 128), mode="nearest", align_corners=None)
        pd = pd.cpu().long()
        gt = torch.from_numpy(curGT_np)[None, None, ...] 
        dice_case = dice_fun(pd, gt).numpy()[0,0]
        hd95_case = hd95_fun(pd, gt).numpy()[0,0]
        nsd_case = nsd_fun(pd, gt).numpy()[0,0]
        asd_case = asd_fun(pd, gt).numpy()[0,0]
        print(f"case {caseID} dice: {dice_case}")

        ## save each case results to txt
        with open(os.path.join(save_eval_path, "test_each_dice_hd95_nsd_asd_rand"+str(NUM)+".txt"), "a") as f:
            write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\t\n"
            f.write(write_content)

        dice_all.append(dice_case)
        hd95_all.append(hd95_case)
        nsd_all.append(nsd_case)
        asd_all.append(asd_case)
        
        # ## save results
        # curMR_np = curMR.permute((0,1,4,2,3)).cpu().numpy()[0,0,...] #DHW3-->DHW .transpose((1,2,0,3)
        # pd = pd.numpy()[0,0,...] # HWD
        # pd = np.transpose(pd, axes=(2, 0, 1))
        # if args.visualize:
        #     seg_sitk = sitk.GetImageFromArray(pd)
        #     seg_sitk.SetSpacing((0.89, 0.89, 3))
        #     sitk.WriteImage(seg_sitk, join(pred_save_dir, caseID+"_pd_sam2.nii.gz"))

        #     img_sitk = sitk.GetImageFromArray(curMR_np)
        #     img_sitk.SetSpacing((0.89, 0.89, 3))
        #     sitk.WriteImage(img_sitk, join(pred_save_dir, caseID+"_im_sam2.nii.gz"))
        #     gts_sitk = sitk.GetImageFromArray(curGT_np)
        #     gts_sitk.SetSpacing((0.89, 0.89, 3))
        #     sitk.WriteImage(gts_sitk, join(pred_save_dir, caseID+"_gt_sam2.nii.gz"))


    print(f"mean dice: {np.array(dice_all).mean()}")
    # ## save results
    # with open(os.path.join(save_eval_path, "test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
    #     write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
    #     write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
    #                                     f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
    #                                     f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
    #                                     f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
    #     f.write(write_content)

    # ## save results
    # with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
    #     write_content = f"{'#'*60}\nmean dice\tmean nsd\tmean hd95\tmean asd\t\n"
    #     write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.4f}\t" \
    #                                     f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
    #                                     f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.4f}\t" \
    #                                     f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
    #     write_content = write_content + f"{np.array(dice_all).mean():.2f}±{np.array(dice_all).std():.2f}\t" \
    #                                     f"{np.array(nsd_all).mean():.2f}±{np.array(nsd_all).std():.2f}\t" \
    #                                     f"{np.array(hd95_all).mean():.2f}±{np.array(hd95_all).std():.2f}\t" \
    #                                     f"{np.array(asd_all).mean():.2f}±{np.array(asd_all).std():.2f}\t\n"
    #     write_content = write_content + f"{np.array(dice_all).mean()*100:.2f}\t" \
    #                                     f"{np.array(nsd_all).mean():.2f}\t" \
    #                                     f"{np.array(hd95_all).mean()*100:.2f}\t" \
    #                                     f"{np.array(asd_all).mean():.2f}\n"
    #     f.write(write_content)
