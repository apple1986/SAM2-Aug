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
# from radioa.model.MedSAM import load_medsam, MedSAMInferer


#####################################
from radioa.utils.SAMMed2D_segment_anything import sam_model_registry as registry_sammed2d
from argparse import Namespace
def load_sammed2d(checkpoint_path, image_size=256, device="cuda"):
    args = Namespace()
    args.image_size = image_size
    args.encoder_adapter = True
    args.sam_checkpoint = checkpoint_path
    model = registry_sammed2d["vit_b"](args).to(device)
    model.eval()

    return model

class SAM_Infer():
    def __init__(self, checkpoint_path, device):
        super(SAM_Infer, self).__init__()
        # self.logit_threshold = 0.5
        # self.D, self.H, self.W = None, None, None
        self.checkpoint_path = checkpoint_path
        self.device = device
        

    def load_model(self, checkpoint_path, device):
        return load_sammed2d(checkpoint_path=checkpoint_path, image_size=256, device=device)    

    @torch.no_grad()
    def segment(self, model, points, box, mask, image_embedding):
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=box,
            masks=mask,
        )

        low_res_masks, iou_predictions  = model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe= model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )

        # Obtain the best mask (measured by predicted iou) from the 3 returned masks
        iou_predictions = iou_predictions[0]  # Indexing within batch : we have batch size 1
        max_value, max_index = torch.max(iou_predictions, dim=0)
        low_res_logits = low_res_masks[0, max_index]

        low_res_pred = torch.sigmoid(low_res_logits)  # (256, 256)


        # max_values, max_indexs = torch.max(iou_predictions, dim=1)
        # max_values = max_values.unsqueeze(1)
        # iou_predictions = max_values
        # low_res_masks = low_res_masks[:, max_indexs]

        return low_res_pred


    def sam_predict(self, img, bbox, H, W):
        """
        img: img_256_tensor

        """
        ## load model
        model = self.load_model(self.checkpoint_path, self.device)
        model = model.eval()
        self.pixel_mean, self.pixel_std = (
        model.pixel_mean.unsqueeze(0),
        model.pixel_std.unsqueeze(0),
        )
        ## normalize
        img = (img - self.pixel_mean) / self.pixel_std 

        # get the image embedding
        with torch.no_grad():
            image_embedding = model.image_encoder(img) # (1, 256, 64, 64)
            bbox = torch.from_numpy(bbox).float().unsqueeze(0).unsqueeze(0).to(self.device)
            low_res_pred = self.segment(model=model, points=None, box=bbox, mask=None, image_embedding=image_embedding)

        low_res_pred = F.interpolate(
            low_res_pred[None, None, ...],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (512, 512)
        slice_pd = (low_res_pred > 0.5).astype(np.uint8)
        return slice_pd



#######################################
## set path
model_names = ["sam-med2d_b.pth",]
checkpoint_path = os.path.join(args.sam_root_path, model_names[0])
# model = load_sam(checkpoint_path, device="cuda")
# print(model)



## testing
image_size = 1024
args.visualize = True
data_root = args.data_root
# bbox_shift = args.bbox_shift
args.bbox_shift = 5
args.noMask = True
device = "cuda:1"  # args.device
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
args.ch3 = True
# MedSAM2-UTSW-GP-Prompt-Mask-20250310-2015     MedSAM2-UTSW-GP-Prompt-Mask-20250310-1937   MedSAM2-UTSW-GP-FT-All-20250311-0112
flag_fix = False # False True

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

NUM = 5
split_types = ["itv", ] # "alltv" "gtv", "ptv",
for split_tyep in split_types:
    model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir/paper_used_model/Sam2_Tiny_Promp_Aug_3Ch/AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5"
    medsam2_checkpoint = os.path.join(model_path, "medsam_model_best_dice.pth") # Ori: sam2_hiera_tiny.pt medsam_model_best_dice.pth
    num_workers = args.num_workers
    root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2"
    save_eval_path = os.path.join(model_path, "dataset_hr/sam-med2d_b_eval_result_"+split_tyep)
    makedirs(save_eval_path, exist_ok=True)
    # save segmentation results
    pred_save_dir = os.path.join(model_path, "dataset_hr/sam-med2d_b_seg_result_"+split_tyep)
    makedirs(pred_save_dir, exist_ok=True)
    args.bbox_shift =  NUM # NUM

    # load fine-tuned model
    model_infer = SAM_Infer(checkpoint_path=checkpoint_path, device=device)
    sam2_transforms = SAM2Transforms(resolution=256, mask_threshold=0)

    ## load data
    all_test_transform = transforms.Compose([
            # CenterCrop(patch_size),
            ToTensor(),
    ])
    testLoader = load_data(root_path, all_test_transform, batch_size=1, split="all", tv_type=split_tyep)

    # load evaluation function
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    cnt_case = 0
    for data_four in testLoader:
        cnt_case = cnt_case + 1
        print(f"cnt case {cnt_case}")
        curMR, preMR, preGT, curGT = data_four["curMR"], data_four["preMR"], data_four["preGT"], data_four["curGT"]
        caseID = data_four["caseID"][0]
        curMR = curMR.to(device) # BCHWD
        preMR = preMR.to(device) # BCHWD
        curGT = curGT.to(device) # BHWD
        preGT = preGT.to(device) # B1HWD
        # print(f"curGT shape: {curGT.shape}")
        # print(f"curMR max value: {curMR.max()}")

        B, C, H, W, D = curMR.shape
        curGT_np = torch.permute(curGT, dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
        # preGT_np = torch.permute(preGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
        ## convert to 3 channels
        curMR = torch.cat((curMR, preMR, preGT), dim=1)
        curMR_np = torch.permute(curMR, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3
        # curGT_t = curGT.unsqueeze(1) # B1HWD

        ## inference
        segs_dict = {}
        label_ids = np.unique(curGT_np)[1:]
        ## Simulate 3D box for each organ
        for label_id in label_ids:
            segs_3d_temp = np.zeros_like(curGT_np, dtype=np.uint8)
            marker_data_id = (curGT_np == label_id).astype(np.uint8)
            # marker_data_id = (preGT_np == label_id).astype(np.uint8) # use preGT to extract bbox
            marker_zids, _, _ = np.where(marker_data_id > 0)
            marker_zids = np.sort(np.unique(marker_zids))
            bbox_dict = {} # key: z_index, value: bbox
            for z in marker_zids:
                z_box_check = box_func(marker_data_id[z, :, :], bbox_shift=args.bbox_shift)
                bbox_dict[z] = z_box_check

            ## inference
            for z in list(marker_zids):
                img_2d = curMR_np[z, :, :, :] # HW3
                mask_2d = preGT[..., z].float() # mask_2d = curGT_t[..., z].float() just for checking
                mask_2d = F.interpolate(mask_2d, size=(256, 256), mode="nearest", align_corners=None)
                if args.noMask is True:
                    mask_2d = None # only use box

                if len(img_2d.shape) == 2:
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
                else:
                    img_3c = img_2d
                # convert the shape to (1, 3, 1024, 1024)
                img_256_tensor = sam2_transforms(img_3c)[None, ...].to(device) # 1x3xHxW
                
                ## get bbox
                box_ori = bbox_dict[z]
                ## check if there are more than one bbox in a mask
                if box_ori.ndim > 1:
                    ## inference one by one
                    for num_box in range(box_ori.shape[0]):
                        box_256 = box_ori[num_box,:] / np.array([W, H, W, H]) * 256 ## adjust to 1024x1024 scale
                        img_2d_seg = model_infer.sam_predict(img_256_tensor*255, box_256[None,:], H, W)
                        segs_3d_temp[z, img_2d_seg>0] = 1
                else:
                    box_256 = box_ori / np.array([W, H, W, H]) * 256 ## adjust to 1024x1024 scale
                    img_2d_seg = model_infer.sam_predict(img_256_tensor*255, box_256[None,:], H, W)
                    segs_3d_temp[z, img_2d_seg>0] = 1

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
        with open(os.path.join(save_eval_path, "test_each_dice_hd95_nsd_asd_rand"+str(NUM)+".txt"), "a") as f:
            write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\t\n"
            f.write(write_content)

        dice_all.append(dice_case)
        hd95_all.append(hd95_case)
        nsd_all.append(nsd_case)
        asd_all.append(asd_case)
        
        ## save results
        curMR_np = curMR_np[..., 0] #DHW3-->DHW .transpose((1,2,0,3)
        if args.visualize:
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
    with open(os.path.join(save_eval_path, "test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        f.write(write_content)

    ## save results
    with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean nsd\tmean hd95\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.4f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.2f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}±{np.array(nsd_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.2f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}±{np.array(asd_all).std():.2f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean()*100:.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}\t" \
                                        f"{np.array(hd95_all).mean()*100:.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}\n"
        f.write(write_content)
