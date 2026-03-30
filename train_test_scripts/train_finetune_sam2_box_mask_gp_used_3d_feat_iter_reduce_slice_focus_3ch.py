# -*- coding: utf-8 -*-
"""
finetune sam2 model on medical image data
only finetune the image encoder and mask decoder
freeze the prompt encoder
"""

## %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from sam2.build_sam import build_sam2
from typing import List, Optional, Tuple
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
import cv2
from medpy.metric import dc


## self-definition
from torchvision import transforms
from dataset.ustwlits import ToTensor, CenterCrop
from builders.utswdataset_builder import load_data_sam
from gp_knife.mask_opt import morph_operation
from model_factory.medsam2_var import load_medsam2_3d_apt, load_medsam2_3d_apt_re

# set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(2024)

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6
root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2"
save_eval_path = os.path.join(root_path, "sam_methods/eval_result/sam2")

## set up parser
parser = argparse.ArgumentParser()
## set path
parser.add_argument("-task_name", type=str, default="MedSAM2-UTSW-GP-FT-Temp")
parser.add_argument("-model_cfg", type=str, default="sam2_hiera_t.yaml", help="model config file")
parser.add_argument("-pretrain_model_path", type=str, default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt",)
parser.add_argument("-pred_save_dir", type=str, default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/check_results", help="Path to save the segmentation results",)
parser.add_argument("-work_dir", type=str, default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=30)
parser.add_argument("-batch_size", type=int, default=4)
parser.add_argument("-bbox_shift", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=1)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=6e-6, metavar="LR", help="learning rate (absolute lr) 6e-6")
parser.add_argument("-resume", type=str, default=None, help="Resuming training from checkpoint")
parser.add_argument("-device", type=str, default="cuda:0")
args, unknown = parser.parse_known_args()


## inference to get 3d prediction logits
def get_3d_prediction_vol(curMR, curGT, preGT, boxesGT, feat_cnn, medsam_model):
    medsam_model.eval()
    for param in medsam_model.parameters():
        param.requires_grad = False  # Freeze parameters
    pd_vol = torch.zeros_like(curGT).to(device) #B1HWD
    for num in range(0, curMR.shape[-1]):
        image = curMR[..., num] #Bx1x512x512
        gt2D = curGT[..., num]
        pre_get2D = preGT[..., num]
        boxes = boxesGT[:,num, :] #Bx4

        if gt2D.sum() < 10: # the masks are empty
            continue
        ## resize to 1024x1024
        image = F.interpolate(image, size=(1024, 1024), mode="bilinear", align_corners=True)
        image = torch.cat((image, image, image), dim=1)
        gt2D = F.interpolate(gt2D, size=(256, 256), mode="nearest", align_corners=None)
        pre_get2D = F.interpolate(pre_get2D, size=(256, 256), mode="nearest", align_corners=None)
        boxes = boxes * 2 # original 512, current 1024
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        ## inference
        with torch.no_grad():
        #     # pd_slice = medsam_model(image, boxes_np, pre_get2D, [feat_cnn[0].detach(), feat_cnn[1].detach()]) # 4x1x256x256
            pd_slice = medsam_model(image, boxes_np, pre_get2D, feat_cnn) # 4x1x256x256
        pd_vol[...,num] = F.interpolate(pd_slice, size=(512, 512), mode="bilinear", align_corners=True) # 1x1x512x512  
    pd_vol.requires_grad = True
    return pd_vol


## training
def main(model_save_path, device, args):
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    model_cfg = args.model_cfg
    sam2_checkpoint = args.pretrain_model_path
    feat_extractor_3d, medsam_model = load_medsam2_3d_apt_re(model_cfg, sam2_checkpoint, 
                                                             device=args.device, use_center_crop=args.use_center_crop)
    ## freeze part block
    # for param in medsam_model.sam2_model.image_encoder.parameters():
    #     param.requires_grad = False 
    # for param in medsam_model.sam_prompt_encoder.parameters():
    #     param.requires_grad = False 

    feat_extractor_3d.train()
    medsam_model.train()
    ## print related model information
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    ) 
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  

    img_mask_encdec_params = list(medsam_model.sam2_model.image_encoder.parameters()) + list(
        medsam_model.sam2_model.sam_mask_decoder.parameters()
    )

    # img_mask_encdec_params = list(medsam_model.sam2_model.sam_prompt_encoder.parameters()) + list(
    #     medsam_model.sam2_model.sam_mask_decoder.parameters()
    # )

    # optimizer = torch.optim.AdamW(
    #     img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay # here, we did not update memory bank module
    # )

    optimizer_cnn = torch.optim.AdamW(
        feat_extractor_3d.parameters(), lr=5e-4, weight_decay=args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn, eta_min=1e-6, T_max=args.cnn_epoch)
    optimizer = torch.optim.AdamW([
        {"params": img_mask_encdec_params, "lr": args.lr, "weight_decay": args.weight_decay},  # sam has LR = 1e-6
        {"params": feat_extractor_3d.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}   # Decoder has LR = 0.0005
    ])


    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  
    ## set loss functions
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    ## set hyper-parameters
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    pre_mean_dice = 0
    
    # Define probabilities for numbers 0, 1
    probs = torch.tensor([0.2, 0.8])
    dice_all = []

    ## load dataset
    train_dataloader = load_data_sam(root_path, transform_policy=None, batch_size=args.batch_size, 
                                     split="train", bbox_shift=args.bbox_shift, fix_bbox=args.fix_box)
    val_dataloader = load_data_sam(root_path, transform_policy=None, batch_size=args.batch_size, 
                                   split="val", bbox_shift=args.bbox_shift, fix_bbox=args.fix_box)
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])

    ## training adaptor
    for epoch in range(start_epoch, args.cnn_epoch):
        epoch_loss = 0
        ## training 3D feature module
        for step, batch_sample in enumerate(tqdm(train_dataloader)):
            curMR, preMR, preGT, curGT, boxesGT = batch_sample["curMR"], batch_sample["preMR"], batch_sample["preGT"], batch_sample["curGT"], batch_sample["bboxes"]
            ## boxesGT: BxDx4
            caseID = batch_sample["caseID"][0]
            curMR = curMR.to(device) # B1HWD: Bx1x512x512x128
            curGT = curGT.to(device) # B1HWD
            preMR = preMR.to(device) # B1HWD
            preGT = preGT.to(device) # B1HWD

            all_input = torch.cat((curMR, preMR, preGT), dim=1) # Bx3xHxWxD

            # select 8 slices 
            # find a non-zeros slice
            nonzero_mask = curGT.abs().sum(dim=(0, 1, 2, 3)) > 0  # Sum over B, C, H, W
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]  # Get valid D indices
            random_index = nonzero_indices[torch.randint(0, len(nonzero_indices), (1,))].item()
            start_idx, end_idx = random_index-3, random_index + 5 # total 8 slices

            # process the inputting data
            if args.use_center_crop:
                # input Bx3x256x256x8
                all_input = F.interpolate(all_input, size=(256, 256,128), mode="trilinear", align_corners=True)
                all_input = all_input[..., start_idx:end_idx]
                ## extract 3d features for each case
                feat_cnn = feat_extractor_3d(all_input)
            else:
                all_input = all_input[..., start_idx:end_idx]
                ## extract 3d features for each case
                feat_cnn = feat_extractor_3d(all_input)
            # get 3D prediction results
            curMR = curMR[..., start_idx:end_idx]
            curGT = curGT[..., start_idx:end_idx]
            preGT = preGT[..., start_idx:end_idx]
            boxesGT = boxesGT[:,start_idx:end_idx,:] #BxDx4

            pd_vol = get_3d_prediction_vol(curMR, curGT, preGT, boxesGT, feat_cnn, medsam_model)
            loss_cnn = seg_loss(pd_vol, curGT) + ce_loss(pd_vol, curGT.float())  
            # print(feat_cnn[0].requires_grad)  # Should be True
            # print(pd_vol.requires_grad)    # Should be True
            # print(loss_cnn.requires_grad)  # Should be True
            loss_cnn.backward()
            optimizer_cnn.step()
            optimizer_cnn.zero_grad()
            epoch_loss += loss_cnn.item()
        
        lr_scheduler.step() # update learning rate after an epoch
        epoch_loss /= step
        print(
            f'Train 3D-CNN, Epoch: {epoch}, Loss: {epoch_loss}'
        )  
        with open(os.path.join(model_save_path, "train_loss_record.txt"), "a") as f:
            write_content = f'CNN Train: Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}\n'
            f.write(write_content)

            
    medsam_model.train()
    for param in medsam_model.parameters():
        param.requires_grad = True  # unFreeze parameters
    
    ## training
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, batch_sample in enumerate(tqdm(train_dataloader)):
            curMR, preMR, preGT, curGT, boxesGT = batch_sample["curMR"], batch_sample["preMR"], batch_sample["preGT"], batch_sample["curGT"], batch_sample["bboxes"]
            caseID = batch_sample["caseID"][0]
            curMR = curMR.to(device) # B1HWD: Bx1x512x512x128
            curGT = curGT.to(device) # B1HWD
            preMR = preMR.to(device) # B1HWD
            preGT = preGT.to(device) # B1HWD
            all_input = torch.cat((curMR, preMR, preGT), dim=1)

            if args.use_center_crop:
                # input Bx3x256x256x8
                all_input = F.interpolate(all_input, size=(256, 256,128), mode="trilinear", align_corners=True)

            for num in range(0, curMR.shape[-1]):
                image = curMR[..., num]
                gt2D = curGT[..., num]
                pre_get2D = preGT[..., num]
                boxes = boxesGT[:,num, :] #Bx4

                if gt2D.sum() < 10: # the masks are empty
                    continue
                ## resize to 1024x1024
                image = F.interpolate(image, size=(1024, 1024), mode="bilinear", align_corners=True)
                image = torch.cat((image, image, image), dim=1)
                gt2D = F.interpolate(gt2D, size=(256, 256), mode="nearest", align_corners=None)
                pre_get2D = F.interpolate(pre_get2D, size=(256, 256), mode="nearest", align_corners=None)
                boxes = boxes * 2 # original 512, current 1024

                optimizer.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                # there is 80% to dropout mask
                # Sample a number based on these probabilities
                # random sample mask
                rand_num = torch.multinomial(probs, num_samples=1).item() # return a num in [0, 1]
                if rand_num > 0:
                    # dropout a slice randomly
                    SEL_NUM = torch.randint(low=0, high=pre_get2D.shape[0], size=(1,)).numpy()[0]
                    pre_get2D[SEL_NUM, ...] = torch.zeros_like(pre_get2D[SEL_NUM, ...], device=device)
                    ## morph operation
                    if 0 <= torch.rand(1).item() < 0.3:
                        pre_get2D = morph_operation(pre_get2D, operation="dilate", kernel_size=3, iterations=1)
                    elif 0.3 <= torch.rand(1).item() < 0.6:
                        pre_get2D = morph_operation(pre_get2D, operation="erode", kernel_size=3, iterations=1)
                else:
                    ## morph operation
                    if 0 <= torch.rand(1).item() < 0.3:
                        pre_get2D = morph_operation(pre_get2D, operation="dilate", kernel_size=2, iterations=1)
                    elif 0.3 <= torch.rand(1).item() < 0.6:
                        pre_get2D = morph_operation(pre_get2D, operation="erode", kernel_size=2, iterations=1)
                ## randmly sample bbox
                if args.dropout_box is True:
                    ## randomly dropout bbox
                    # if 0 <= torch.rand(1).item() <= 0.5:
                    # there is 1/bs prob to dropout the bbox
                    # dropout a bbox randomly
                    SEL_NUM = torch.randint(low=0, high=pre_get2D.shape[0], size=(1,)).numpy()[0]
                    # boxes_np[SEL_NUM, ...] = torch.zeros_like(boxes_np[SEL_NUM, ...], device=device)                            
                    boxes_np[SEL_NUM] = np.ones((4))* (-200) # set
                    # medsam_pred = medsam_model(image, boxes_np, pre_get2D)

                ## extract 3d features for near the bbox 0~8 slices
                if num >= 3:
                    feat_cnn = feat_extractor_3d(all_input[..., num-3:num + 5])
                    ## inference
                    medsam_pred = medsam_model(image, boxes_np, pre_get2D, [feat_cnn[0].detach(), feat_cnn[1].detach()])
                else:
                    medsam_pred = medsam_model(image, boxes_np, pre_get2D, feat_cnn=None)
        

                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

        epoch_loss /= step

        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        with open(os.path.join(model_save_path, "train_loss_record.txt"), "a") as f:
            write_content = f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}\n'
            f.write(write_content)
        ## save the current model
        checkpoint = {
            "adaptor_model": feat_extractor_3d.state_dict(),
            "sam2_model": medsam_model.state_dict(),
            }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

        ## do evaluation
        dice_all = []
        for batch_sample in val_dataloader:
            curMR, preMR, preGT, curGT, boxesGT = batch_sample["curMR"], batch_sample["preMR"], batch_sample["preGT"], batch_sample["curGT"], batch_sample["bboxes"]
            caseID = batch_sample["caseID"][0]
            curMR = curMR.to(device) # B1HWD: Bx1x512x512x128
            curGT = curGT.to(device) # B1HWD
            preMR = preMR.to(device) # B1HWD
            preGT = preGT.to(device) # B1HWD
            all_input = torch.cat((curMR, preMR, preGT), dim=1)
            if args.use_center_crop:
                # input Bx3x256x256x8
                all_input = F.interpolate(all_input, size=(256, 256,128), mode="trilinear", align_corners=True)

            pd_vol = torch.zeros_like(curGT).to(device)

            for num in range(0, curMR.shape[-1]):
                image = curMR[..., num]
                gt2D = curGT[..., num]
                pre_get2D = preGT[..., num]
                boxes = boxesGT[:,num, :] #Bx4

                if gt2D.sum() < 10: # the masks are empty
                    continue
                ## resize to 1024x1024
                image = F.interpolate(image, size=(1024, 1024), mode="bilinear", align_corners=True)
                image = torch.cat((image, image, image), dim=1)
                gt2D = F.interpolate(gt2D, size=(256, 256), mode="nearest", align_corners=None)
                pre_get2D = F.interpolate(pre_get2D, size=(256, 256), mode="nearest", align_corners=None)
                boxes = boxes * 2 # original 512, current 1024

                optimizer.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                ## inference
                ## extract 3d features for near the bbox 0~8 slices
                if num >= 3:
                    feat_cnn = feat_extractor_3d(all_input[..., num-3:num + 5])
                    low_res_pred = medsam_model(image, boxes_np, pre_get2D, [feat_cnn[0].detach(), feat_cnn[1].detach()])
                else:
                    low_res_pred = medsam_model(image, boxes_np, pre_get2D, feat_cnn=None)
                medsam_pred = (low_res_pred > 0.5).float()
                medsam_pred = F.interpolate(medsam_pred, size=(512, 512), mode="nearest", align_corners=None) # 1x1x512x512  

                pd_vol[..., num] = medsam_pred
            # calculate dice
            dice_case = dc(pd_vol.detach().cpu().numpy(), curGT.detach().cpu().numpy())
            print(f"case {caseID} dice: {dice_case}")
            dice_all.append(dice_case)
        cur_mean_dice = np.array(dice_all).mean()
        print(f"mean dice: {cur_mean_dice}")

        ## record dice 
        with open(os.path.join(model_save_path, "val_dice_record.txt"), "a") as f:
            write_content = f'Epoch: {epoch}, mean dice: {cur_mean_dice}\n'
            f.write(write_content)        

        ## save the best model
        if cur_mean_dice > pre_mean_dice :
            pre_mean_dice = cur_mean_dice
            checkpoint = {
                "adaptor_model": feat_extractor_3d.state_dict(),
                "sam2_model": medsam_model.state_dict(),
                }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best_dice.pth"))
        
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "adaptor_model": feat_extractor_3d.state_dict(),
                "sam2_model": medsam_model.state_dict(),
                }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best_loss.pth"))

if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    ## set model
    args.model_cfg = "sam2_hiera_t.yaml"
    args.pretrain_model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt"
    args.device = "cuda:0"
    device = torch.device(args.device)
    args.fix_box=False
    args.dropout_box = False
    args.use_center_crop = True
    args.cnn_epoch = 50
    model_save_path = join(args.work_dir, args.task_name) #  + "-" + run_id
    
    for num in range(5, 6): # fix box shiffting: 0, 1, 2, 3, 4, 5, num=0 has been done
        args.task_name = "Sam2_Tiny_Promp_Aug_Shift_5/CNN_Adaptor_FocusCenter_3ch_Rand"+str(num)
        args.bbox_shift = num
        model_save_path = join(args.work_dir, args.task_name)
        main(model_save_path, device, args)


