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
from gp_knife.loss_margin import BCELogitMarginAbs

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
parser.add_argument("-num_epochs", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=4)
parser.add_argument("-bbox_shift", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=1)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=6e-6, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("-resume", type=str, default=None, help="Resuming training from checkpoint")
parser.add_argument("-device", type=str, default="cuda:0")
args, unknown = parser.parse_known_args()



## model compoments
class MedSAM2(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # unfreeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = True
        # unfreeze mask decoder
        for param in self.sam2_model.sam_mask_decoder.parameters():
            param.requires_grad = True
        # freeze image encoder
        for param in self.sam2_model.image_encoder.parameters():
            param.requires_grad = True
        
        # # Memory block
        # for param in self.sam2_model.memory_encoder.parameters():
        #     param.requires_grad = True
        # for param in self.sam2_model.memory_attention.parameters():
        #     param.requires_grad = True

        ## version 1
        self.refine = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                    nn.PReLU(num_parameters=1, init=0.5),
                                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
                                    nn.PReLU(num_parameters=1, init=0.5),)

        # ## version 2
        # self.refine = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=16, kernel_size=3, stride=1, padding=1),
        #                             nn.GELU(),
        #                             nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
        #                             nn.GELU()
        #                             )


            
    def forward(self, image, box, masks=None, idx=None, prePd=None,):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        masks: Bx4
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do compute gradients for prompt encoder
        # with torch.no_grad():
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
        ## ?? ap: what is meaning
        if len(box_torch.shape) == 2:
            box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
            box_labels = box_labels.repeat(box_torch.size(0), 1)
        concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=masks,
        )

        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        ## refine the mask
        # prePd[:, idx:idx+1, :, :] = low_res_masks_logits # BDHW: ([4, 1, 256, 256])
        prePd = prePd.clone()  # Preserve autograd graph
        prePd[:, idx:idx+1, :, :] = torch.sigmoid(low_res_masks_logits)
        low_res_masks_logits_refine = self.refine(prePd) # B1HW
        # pd_refine = self.refine(prePd)
        # low_res_masks_logits_refine = low_res_masks_logits.min() + (low_res_masks_logits.max() - low_res_masks_logits.min()) * \
        #                                     ((pd_refine - pd_refine.min()) / (pd_refine.max() - pd_refine.min())) # B1HW
        low_res_masks_logits = low_res_masks_logits + low_res_masks_logits_refine

        return low_res_masks_logits, prePd
    
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features


## main func
def main(model_save_path, device, args):
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    model_cfg = args.model_cfg
    sam2_checkpoint = args.pretrain_model_path
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=args.device, apply_postprocessing=True)
    medsam_model = MedSAM2(model=sam2_model)
    medsam_model = medsam_model.to(args.device)
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

    if args.ft_img_mask_prompt is True:
        model_params = list(medsam_model.sam2_model.image_encoder.parameters()) + \
                            list(medsam_model.sam2_model.sam_mask_decoder.parameters()) + \
                                            list(medsam_model.sam2_model.sam_prompt_encoder.parameters()) 
        # optimizer = torch.optim.AdamW([
        #     {'params': model_params, "lr":args.lr, "weight_decay":args.weight_decay},
        #     {'params': medsam_model.refine.parameters(), "lr":6e-2, "weight_decay":args.weight_decay},
        # ])
        # scheduler_refine = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs

        optimizer = torch.optim.AdamW(
            model_params, lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer_refine = torch.optim.AdamW(
            medsam_model.refine.parameters(), lr=6e-5, weight_decay=args.weight_decay
        )
        scheduler_refine = torch.optim.lr_scheduler.StepLR(optimizer_refine, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs
        for param in medsam_model.refine.parameters():
             param.requires_grad = True

        print(
            "Number of selected training parameters: ",
            sum(p.numel() for p in model_params if p.requires_grad),
        )  
    elif args.ft_img_mask is True:
        model_params = list(medsam_model.sam2_model.image_encoder.parameters()) + \
                            list(medsam_model.sam2_model.sam_mask_decoder.parameters())
        optimizer = torch.optim.AdamW(
            model_params, lr=args.lr, weight_decay=args.weight_decay
        )
        print(
            "Number of image encoder and mask decoder parameters: ",
            sum(p.numel() for p in model_params if p.requires_grad),
        )  

    ## set loss functions
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    # ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    ce_margin_loss = BCELogitMarginAbs(margin=5)

    ## set hyper-parameters
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    pre_mean_dice = 0
    start_epoch = 0
    # Define probabilities for numbers 0, 1
    probs = torch.tensor([0.2, 0.8])
    dice_all = []
    ## load dataset
    train_dataloader = load_data_sam(root_path, transform_policy=None, batch_size=args.batch_size, 
                                     split="train", bbox_shift=args.bbox_shift, fix_bbox=args.fix_box, pos_neg=args.pos_neg)
    val_dataloader = load_data_sam(root_path, transform_policy=None, batch_size=args.batch_size, 
                                   split="val", bbox_shift=args.bbox_shift, fix_bbox=args.fix_box, pos_neg=args.pos_neg)
    
    ## training
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, batch_sample in enumerate(tqdm(train_dataloader)):
            curMR, preMR, preGT, curGT, boxesGT = batch_sample["curMR"], batch_sample["preMR"], batch_sample["preGT"], batch_sample["curGT"], batch_sample["bboxes"]
            caseID = batch_sample["caseID"][0]
            curMR = curMR.to(device) # B1HWD: Bx1x512x512x128
            preMR = preMR.to(device) # B1HWD: Bx1x512x512x128   
            curGT = curGT.to(device) # B1HWD
            preGT = preGT.to(device) # B1HWD
            if args.use_preGT is True:
                prePd = F.interpolate(preGT, size=(256, 256, 128), mode="nearest")[:,0,...] #BhwD
                prePd = torch.permute(prePd, dims=(0,3,1,2)) #BDhw
            else:
                prePd = torch.zeros((preGT.shape[0], 128, 256, 256)).to(device) # BDhw

            if args.ch3 is True:
                curMR = torch.cat((curMR, preMR, preGT), dim=1)

            for num in range(0, curMR.shape[-1]):
                image = curMR[..., num]
                gt2D = curGT[..., num]
                preMask = preGT[..., num]
                boxes = boxesGT[:,num, :] #Bx4

                if gt2D.sum() < 10: # the masks are empty
                    continue
                ## resize to 1024x1024
                image = F.interpolate(image, size=(1024, 1024), mode="bilinear", align_corners=True)
                if args.ch3 is not True:
                    image = torch.cat((image, image, image), dim=1)
                gt2D = F.interpolate(gt2D, size=(256, 256), mode="nearest", align_corners=None)
                preMask = F.interpolate(preMask, size=(256, 256), mode="nearest", align_corners=None) # mask prompt
                preMask_ori = preMask #.clone()
                boxes = boxes * 2 # original 512, current 1024

                optimizer.zero_grad()
                optimizer_refine.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                
                ## inference
                # there is 80% to dropout mask
                # Sample a number based on these probabilities
                # random sample mask
                rand_num = torch.multinomial(probs, num_samples=1).item() # return a num in [0, 1]
                if rand_num > 0:
                    # dropout a slice randomly
                    SEL_NUM = torch.randint(low=0, high=preMask.shape[0], size=(1,)).numpy()[0]
                    preMask[SEL_NUM, ...] = torch.zeros_like(preMask[SEL_NUM, ...], device=device)
                    ## morph operation
                    if 0 <= torch.rand(1).item() < 0.3:
                        preMask = morph_operation(preMask, operation="dilate", kernel_size=3, iterations=1)
                    elif 0.3 <= torch.rand(1).item() < 0.6:
                        preMask = morph_operation(preMask, operation="erode", kernel_size=3, iterations=1)
                else:
                    ## morph operation
                    if 0 <= torch.rand(1).item() < 0.3:
                        preMask = morph_operation(preMask, operation="dilate", kernel_size=2, iterations=1)
                    elif 0.3 <= torch.rand(1).item() < 0.6:
                        preMask = morph_operation(preMask, operation="erode", kernel_size=2, iterations=1)
                ## randmly sample bbox
                if args.dropout_box is True:
                    ## randomly dropout bbox
                    if 0 <= torch.rand(1).item() <= 0.5:
                        # # there is 1/bs prob to dropout the bbox
                        # SEL_NUM = torch.randint(low=0, high=preMask.shape[0], size=(1,)).numpy()[0]                        
                        # boxes_np[SEL_NUM] = np.ones((4)) * (-200) # set
                        # dropout a batch of bbox randomly
                        boxes_np = np.ones_like(boxes_np) * (-200) # dropout all box
                ## inference
                if args.use_mask is not True:
                    preMask = None # do not use mask as prompt
                if (args.use_aug is not True) and (args.use_mask is True):
                    preMask = preMask_ori
                medsam_pred, prePd = medsam_model(image, boxes_np, preMask, idx=num, prePd=prePd)
                ## calculate loss
                # loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss = seg_loss(medsam_pred, gt2D) + ce_margin_loss(medsam_pred, gt2D.float()) / 10.0
                # print(f"diceLoss:{seg_loss(medsam_pred, gt2D):.2f}, bceLoss:{ce_margin_loss(medsam_pred, gt2D.float()):.2f}, ceLoss:{ce_loss(medsam_pred, gt2D.float()):.4f}")

                ## update            
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                optimizer_refine.step()
                optimizer_refine.zero_grad()
                epoch_loss += loss.item()
                prePd = prePd.detach()

        ## update learning rate
        scheduler_refine.step()
        epoch_loss /= step
        losses.append(epoch_loss)
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        print(f"learning rate of refine module: {scheduler_refine.get_last_lr()}")
        with open(os.path.join(model_save_path, "train_loss_record.txt"), "a") as f:
            write_content = f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}\n'
            f.write(write_content)
        ## save the current model
        checkpoint = medsam_model.state_dict()
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

        ## do evaluation
        dice_all = []
        for batch_sample in val_dataloader:
            curMR, preMR, preGT, curGT, boxesGT = batch_sample["curMR"], batch_sample["preMR"], batch_sample["preGT"], batch_sample["curGT"], batch_sample["bboxes"]
            caseID = batch_sample["caseID"][0]
            curMR = curMR.to(device) # B1HWD: Bx1x512x512x128
            preMR = preMR.to(device)
            curGT = curGT.to(device) # B1HWD
            preGT = preGT.to(device) # B1HWD
            pd_vol = torch.zeros_like(curGT).to(device)
            if args.ch3 is True:
                curMR = torch.cat((curMR, preMR, preGT), dim=1)
            
            if args.use_preGT is True:
                prePd = F.interpolate(preGT, size=(256, 256, 128), mode="nearest")[:,0,...] #BhwD
                prePd = torch.permute(prePd, dims=(0,3,1,2)) #BDhw
            else:
                prePd = torch.zeros((preGT.shape[0], 128, 256, 256)).to(device) # BDhw

            for num in range(0, curMR.shape[-1]):
                image = curMR[..., num]
                gt2D = curGT[..., num]
                preMask = preGT[..., num]
                boxes = boxesGT[:,num, :] #Bx4

                if gt2D.sum() < 10: # the masks are empty
                    continue
                ## resize to 1024x1024
                image = F.interpolate(image, size=(1024, 1024), mode="bilinear", align_corners=True)
                if args.ch3 is not True:
                    image = torch.cat((image, image, image), dim=1)
                gt2D = F.interpolate(gt2D, size=(256, 256), mode="nearest", align_corners=None)
                preMask = F.interpolate(preMask, size=(256, 256), mode="nearest", align_corners=None)
                boxes = boxes * 2 # original 512, current 1024

                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                ## inference
                if args.use_mask is not True:
                    preMask = None # do not use mask as prompt
                # low_res_pred = medsam_model(image, boxes_np, preMask)
                with torch.no_grad():
                    low_res_pred, prePd = medsam_model(image, boxes_np, preMask, idx=num, prePd=prePd)
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
        if cur_mean_dice > pre_mean_dice:
            pre_mean_dice = cur_mean_dice
            checkpoint = medsam_model.state_dict()
            torch.save(checkpoint, join(model_save_path, "medsam_model_best_dice.pth"))
        
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = medsam_model.state_dict()
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
    args.ft_img_mask = False
    args.ft_img_mask_prompt = True # finetuning image encoder, mask decoder and prompt
    args.ch3 = True
    args.pos_neg = False
    args.use_mask = True
    args.use_aug = True
    args.use_preGT = False
    model_save_path = join(args.work_dir, args.task_name) #  + "-" + run_id
    
    for num in range(5, 6): # fix box shiffting: 0, 1, 2, 3, 4, 5, num=0 has been done
        args.task_name = "Sam2_Tiny_Promp_Aug_3Ch_3D_Infor/FT_Img_Msk_Prmp_3ch_MargLoss_Tmp2_Rand"+str(num)
        args.bbox_shift = num 
        model_save_path = join(args.work_dir, args.task_name)
        main(model_save_path, device, args)


