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
from model_factory.medsam2_var import load_medsam2_3d_apt

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
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=4)
parser.add_argument("-bbox_shift", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=1)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=6e-7, metavar="LR", help="learning rate (absolute lr) 6e-6")
parser.add_argument("-resume", type=str, default=None, help="Resuming training from checkpoint")
parser.add_argument("-device", type=str, default="cuda:0")
args, unknown = parser.parse_known_args()



##
def main(model_save_path, device, args):
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    model_cfg = args.model_cfg
    sam2_checkpoint = args.pretrain_model_path
    feat_extractor_3d, medsam_model = load_medsam2_3d_apt(model_cfg, sam2_checkpoint, device=args.device)
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
    # optimizer = torch.optim.AdamW(
    #     img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay # here, we did not update memory bank module
    # )
    # optimizer_cnn = torch.optim.AdamW(
    #     feat_extractor_3d.parameters(), lr=1e-4, weight_decay=args.weight_decay,
    # )

    optimizer = torch.optim.AdamW([
        {"params": img_mask_encdec_params, "lr": args.lr, "weight_decay": args.weight_decay},  # sam has LR = 1e-6
        {"params": feat_extractor_3d.parameters(), "lr": 1e-3, "weight_decay": args.weight_decay}   # Decoder has LR = 0.0005
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
    flag_gt = "preGT"
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

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, batch_sample in enumerate(tqdm(train_dataloader)):
            curMR, preMR, preGT, curGT, boxesGT = batch_sample["curMR"], batch_sample["preMR"], batch_sample["preGT"], batch_sample["curGT"], batch_sample["bboxes"]
            caseID = batch_sample["caseID"][0]
            curMR = curMR.to(device) # B1HWD: Bx1x512x512x128
            curGT = curGT.to(device) # B1HWD
            preMR = preMR.to(device)
            preGT = preGT.to(device) # B1HWD

            ## extract 3d features for each case
            feat_cnn = feat_extractor_3d(torch.cat((curMR, preMR, preGT), dim=1))

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
                # optimizer_cnn.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                if "preGT" in flag_gt:
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

                    ## inference
                    medsam_pred = medsam_model(image, boxes_np, pre_get2D, [feat_cnn[0].detach(), feat_cnn[1].detach()])
                    # medsam_pred = medsam_model(image, boxes_np, pre_get2D, feat_cnn=None)
                else:
                    # there is 50% to do this
                    # Sample a number based on these probabilities
                    rand_num = torch.multinomial(probs, num_samples=1).item()
                    if rand_num > 0:
                        # dropout a slice randomly
                        SEL_NUM = torch.randint(low=0, high=gt2D.shape[0], size=(1,)).numpy()[0]
                        gt2D[SEL_NUM, ...] = torch.zeros_like(gt2D[SEL_NUM, ...], device=device)
                        ## morph operation
                        if 0 <= torch.rand(1).item() < 0.3:
                            gt2D = morph_operation(gt2D, operation="dilate", kernel_size=2, iterations=1)
                        elif 0.3 <= torch.rand(1).item() < 0.6:
                            gt2D = morph_operation(gt2D, operation="erode", kernel_size=2, iterations=1)
                        medsam_pred = medsam_model(image, boxes_np, gt2D)
                    else:
                        ## morph operation
                        if 0 <= torch.rand(1).item() < 0.3:
                            gt2D = morph_operation(gt2D, operation="dilate", kernel_size=2, iterations=1)
                        elif 0.3 <= torch.rand(1).item() < 0.6:
                            gt2D = morph_operation(gt2D, operation="erode", kernel_size=2, iterations=1)
                        medsam_pred = medsam_model(image, boxes_np, gt2D)

                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())              
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # optimizer_cnn.step()
                # optimizer_cnn.zero_grad()
                epoch_loss += loss.item()

        epoch_loss /= step
        losses.append(epoch_loss)
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
            pd_vol = torch.zeros_like(curGT).to(device)

            ## extract 3d features for each case
            feat_cnn = feat_extractor_3d(torch.cat((curMR, preMR, preGT), dim=1))

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
                # low_res_pred = medsam_model(image, boxes_np, pre_get2D)
                low_res_pred = medsam_model(image, boxes_np, pre_get2D, [feat_cnn[0].detach(), feat_cnn[1].detach()])
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
    args.device = "cuda:1"
    device = torch.device(args.device)
    args.fix_box=False
    args.dropout_box = True
    model_save_path = join(args.work_dir, args.task_name) #  + "-" + run_id
    
    for num in range(3, 4): # fix box shiffting: 0, 1, 2, 3, 4, 5, num=0 has been done
        args.task_name = "Sam2_Tiny_BBox_Dropout/MedSAM2-UTSW-GP-FT-ALL-3D_CNN_Adaptor_Rand"+str(num)
        args.bbox_shift = num
        model_save_path = join(args.work_dir, args.task_name)
        main(model_save_path, device, args)


