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
from gp_knife.eval_model import get_dice_hd_nsd_asd
from model_factory.medsam2_var import load_medsam2_3d_apt, load_medsam2_3d_apt_re
from builders.utswdataset_builder import load_data_sam


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

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


#######################################
## testing
image_size = 1024
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
visualize = args.visualize
data_root = args.data_root
# bbox_shift = args.bbox_shift
args.bbox_shift = 0
device = "cuda:1"  # args.device
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
args.reduce_slice = True
# MedSAM2-UTSW-GP-Prompt-Mask-20250310-2015     MedSAM2-UTSW-GP-Prompt-Mask-20250310-1937   MedSAM2-UTSW-GP-FT-All-20250311-0112
flag_fix = True # False True

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

for NUM in range(0, 3):
    model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir/Sam2_Tiny_BBox_Dropout/MedSAM2-UTSW-GP-FT-Mask-3D_CNN_Adaptor_Reduce16slices_Iter_Rand"+str(NUM)
    # model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir/Sam2_Tiny_BBox_Shift_Zero/MedSAM2-UTSW-GP-FT-ALL-Box-Mask-20250312-0031"
    medsam2_checkpoint = os.path.join(model_path, "medsam_model_best_dice.pth") # Ori: sam2_hiera_tiny.pt medsam_model_best_dice.pth
    num_workers = args.num_workers
    root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2"
    save_eval_path = os.path.join(model_path, "eval_result_check")
    makedirs(save_eval_path, exist_ok=True)
    # save segmentation results
    pred_save_dir = os.path.join(model_path, "seg_result_check")
    makedirs(pred_save_dir, exist_ok=True)
    args.bbox_shift = NUM

    # load fine-tuned model
    if args.reduce_slice is True:
        feat_extractor_3d, sam2_model = load_medsam2_3d_apt_re(model_cfg, sam2_checkpoint, device=device)
    else:
        feat_extractor_3d, sam2_model = load_medsam2_3d_apt(model_cfg, sam2_checkpoint, device=device)
    checkpoint = torch.load(medsam2_checkpoint, map_location=device if torch.cuda.is_available() else "cpu")
    feat_extractor_3d.load_state_dict(checkpoint["adaptor_model"])
    feat_extractor_3d.eval()
    sam2_model.load_state_dict(checkpoint, strict=False)
    sam2_model.eval()
    sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)

    ## load data
    all_test_transform = transforms.Compose([
            # CenterCrop(patch_size),
            ToTensor(),
    ])
    testLoader = load_data_sam(root_path, transform_policy=None, batch_size=1, 
                                     split="val", bbox_shift=1, fix_bbox=False)
    # load evaluation function
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    for batch_sample in testLoader:
        curMR, preMR, preGT, curGT, boxesGT = batch_sample["curMR"], batch_sample["preMR"], batch_sample["preGT"], batch_sample["curGT"], batch_sample["bboxes"]
        caseID = batch_sample["caseID"][0]
        curMR = curMR.to(device) # BCHWD
        curGT = curGT.to(device) # B1HWD
        preMR = preMR.to(device) # B1HWD
        preGT = preGT.to(device) # B1HWD
        all_input = torch.cat((curMR, preMR, preGT), dim=1)
        # print(f"curGT shape: {curGT.shape}")
        # print(f"curMR max value: {curMR.max()}")
        B, C, H, W, D = curMR.shape # 1x1x512x512x128
        curMR_np = torch.permute(curMR, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
        curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
        # preGT_np = torch.permute(preGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
        
        if args.reduce_slice is False:
            ## inference
            ## extract 3d features
            feat_cnn = feat_extractor_3d(torch.cat((curMR, preMR, preGT), dim=1))
        # feat_cnn = None
        ## test each slice one by one
        segs_dict = {}
        label_ids = np.unique(curGT_np)[1:]
        ## Simulate 3D box for each lesion
        for label_id in label_ids:
            segs_3d_temp = np.zeros_like(curMR_np, dtype=np.uint8)
            marker_data_id = (curGT_np == label_id).astype(np.uint8)
            marker_zids, _, _ = np.where(marker_data_id > 0)
            marker_zids = np.sort(np.unique(marker_zids))
            bbox_dict = {} # key: z_index, value: bbox
            for z in marker_zids:
                z_box_check = box_func(marker_data_id[z, :, :], bbox_shift=args.bbox_shift)
                bbox_dict[z] = z_box_check

            ## inference
            for z in list(marker_zids):
                img_2d = curMR_np[z, :, :] # HxW
                img_2d = img_2d[None, None] #1x1xHxW
                mask_2d = preGT[..., z].float() 
                mask_2d = F.interpolate(mask_2d, size=(256, 256), mode="nearest", align_corners=None)
                
                ## resize to 1024x1024
                image = F.interpolate(torch.from_numpy(img_2d), size=(1024, 1024), mode="bilinear", align_corners=True)
                img_1024_tensor = torch.cat((image, image, image), dim=1).to(device)

                ## get bbox and inference
                box_ori = bbox_dict[z]
                ## check if there are more than one bbox in a mask
                if box_ori.ndim > 1:
                    ## inference one by one
                    for num_box in range(box_ori.shape[0]):
                        box_1024 = box_ori[num_box,:] / np.array([W, H, W, H]) * 1024 ## adjust to 1024x1024 scale
                        with torch.no_grad(): 
                            if args.reduce_slice is True and (num_box >= 3):
                                ## inference
                                ## extract 3d features
                                feat_cnn = feat_extractor_3d(all_input[..., z-3:z+5])
                            # else:
                            #     feat_cnn=None
                            low_res_pred = sam2_model(img_1024_tensor, box_1024[None, :], mask_2d, feat_cnn)
                        # threshold
                        medsam_pred = (low_res_pred > 0.5).float()  
                        medsam_pred = F.interpolate(medsam_pred, size=(512, 512), mode="nearest", align_corners=None) # 1x1x512x512 
                        img_2d_seg =  medsam_pred.cpu().numpy()[0,0,...] 
                        segs_3d_temp[z, img_2d_seg>0] = 1
                else:
                    box_1024 = box_ori / np.array([W, H, W, H]) * 1024 ## adjust to 1024x1024 scale
                    with torch.no_grad():
                        if args.reduce_slice is True and (z >= 3):
                            ## inference
                            ## extract 3d features
                            feat_cnn = feat_extractor_3d(all_input[..., z-3:z+5])
                        # else:
                        #     feat_cnn=None
                        low_res_pred = sam2_model(img_1024_tensor, box_1024[None, :], mask_2d, feat_cnn)
                    # threshold
                    medsam_pred = (low_res_pred > 0.5).float()  
                    medsam_pred = F.interpolate(medsam_pred, size=(512, 512), mode="nearest", align_corners=None) # 1x1x512x512 
                    img_2d_seg =  medsam_pred.cpu().numpy()[0,0,...] 
                    segs_3d_temp[z, img_2d_seg>0] = 1 #DHW

            segs_dict[label_id] = segs_3d_temp.copy()
        # cal dice, hd95, nsd and asd
        pd = torch.from_numpy(segs_3d_temp)[None, None, ...] #BCDHW
        gt = torch.from_numpy(curGT_np)[None, None, ...] 
        dice_case = dice_fun(pd, gt).numpy()[0,0]
        hd95_case = hd95_fun(pd, gt).numpy()[0,0]
        nsd_case = nsd_fun(pd, gt).numpy()[0,0]
        asd_case = asd_fun(pd, gt).numpy()[0,0]
        print(f"case {caseID} dice: {dice_case}")

        ## save each case results to txt
        with open(os.path.join(save_eval_path, "test_each_dice_hd95_nsd_asd.txt"), "a") as f:
            write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\t\n"
            f.write(write_content)

        dice_all.append(dice_case)
        hd95_all.append(hd95_case)
        nsd_all.append(nsd_case)
        asd_all.append(asd_case)
        
        ## save results
        if visualize:
            for label_id in label_ids:
                seg_sitk = sitk.GetImageFromArray(segs_dict[label_id])
                seg_sitk.SetSpacing((0.89, 0.89, 3))
                sitk.WriteImage(seg_sitk, join(pred_save_dir, caseID+"_pd_sam2.nii.gz"))

            img_sitk = sitk.GetImageFromArray(curMR_np)
            img_sitk.SetSpacing((0.89, 0.89, 3))
            sitk.WriteImage(img_sitk, join(pred_save_dir, caseID+"_im_sam2.nii.gz"))
            gts_sitk = sitk.GetImageFromArray(curGT_np)
            gts_sitk.SetSpacing((0.89, 0.89, 3))
            sitk.WriteImage(gts_sitk, join(pred_save_dir, caseID+"_gt_sam2.nii.gz"))


    print(f"mean dice: {np.array(dice_all).mean()}")
    ## save results
    with open(os.path.join(save_eval_path, "test_mean_dice_hd_nsd_asd.txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        f.write(write_content)

