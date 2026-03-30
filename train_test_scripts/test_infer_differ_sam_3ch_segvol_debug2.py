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
import argparse
import os
import torch
import torch.nn.functional as F
import json
from segment_anything_volumetric import sam_model_registry
from network.model import SegVol
from data_process.demo_data_process import process_ct_gt
import monai.transforms as transforms
from utils.monai_inferers_utils import sliding_window_inference, generate_box, select_points, build_binary_cube, build_binary_points, logits2roi_coor
from utils.visualize import draw_result

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
###########################
## set parameters
def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", default=True, type=bool)
    parser.add_argument("--resume", type = str, default = '')
    parser.add_argument("-infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    ### demo
    parser.add_argument('--demo_config', type=str, required=True)
    parser.add_argument("--clip_ckpt", type = str, default = './config/clip')

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
    return args



## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

#######################################
"""different sam-based models"""
def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match\n" + str(preds.shape) + str(labels.shape)
    predict = preds.view(1, -1)
    target = labels.view(1, -1)
    if target.shape[1] < 1e8:
        predict = predict.cuda()
        target = target.cuda()
    predict = torch.sigmoid(predict)
    predict = torch.where(predict > 0.5, 1., 0.)
    
    tp = torch.sum(torch.mul(predict, target))
    den = torch.sum(predict) + torch.sum(target) + 1
    dice = 2 * tp / den

    if target.shape[1] < 1e8:
        predict = predict.cpu()
        target = target.cpu()
    return dice

def zoom_in_zoom_out(args, segvol_model, image, image_resize, gt3D, gt3D_resize, categories=None):
    logits_labels_record = {}
    image_single_resize = image_resize
    image_single = image[0,0]
    ori_shape = image_single.shape
    for item_idx in range(len(categories)):
        # get label to generate prompts
        label_single = gt3D[0][item_idx]
        label_single_resize = gt3D_resize[0][item_idx]
        # skip meaningless categories
        if torch.sum(label_single) == 0:
            print('No object, skip')
            continue
        # generate prompts
        text_single = categories[item_idx] if args.use_text_prompt else None
        if categories is not None: print(f'inference |{categories[item_idx]}| target...')
        points_single = None
        box_single = None
        if args.use_point_prompt:
            point, point_label = select_points(label_single_resize, num_positive_extra=3, num_negative_extra=3)
            points_single = (point.unsqueeze(0).float().cuda(), point_label.unsqueeze(0).float().cuda()) 
            binary_points_resize = build_binary_points(point, point_label, label_single_resize.shape)
        if args.use_box_prompt:
            box_single = generate_box(label_single_resize).unsqueeze(0).float().cuda()
            binary_cube_resize = build_binary_cube(box_single, binary_cube_shape=label_single_resize.shape)
        
        ####################
        # zoom-out inference:
        print('--- zoom out inference ---')
        print(f'use text-prompt [{text_single!=None}], use box-prompt [{box_single!=None}], use point-prompt [{points_single!=None}]')
        with torch.no_grad():
            logits_global_single = segvol_model(image_single_resize.cuda(),
                                                text=text_single, 
                                                boxes=box_single, 
                                                points=points_single)
        
        # resize back global logits
        logits_global_single = F.interpolate(
                logits_global_single.cpu(),
                size=ori_shape, mode='nearest')[0][0]
        
        # build prompt reflection for zoom-in
        if args.use_point_prompt:
            binary_points = F.interpolate(
                binary_points_resize.unsqueeze(0).unsqueeze(0).float(),
                size=ori_shape, mode='nearest')[0][0]
        if args.use_box_prompt:
            binary_cube = F.interpolate(
                binary_cube_resize.unsqueeze(0).unsqueeze(0).float(),
                size=ori_shape, mode='nearest')[0][0]
        zoom_out_dice = dice_score(logits_global_single.squeeze(), label_single.squeeze())
        logits_labels_record[categories[item_idx]] = (
            zoom_out_dice,
            image_single, 
            points_single,
            box_single,
            logits_global_single, 
            label_single)
        print(f'zoom out inference done with zoom_out_dice: {zoom_out_dice:.4f}')
        if not args.use_zoom_in:
            continue

        ####################
        # zoom-in inference:
        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(args.spatial_size, logits_global_single)
        if min_d is None:
            print('Fail to detect foreground!')
            continue

        # Crop roi
        image_single_cropped = image_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
        global_preds = (torch.sigmoid(logits_global_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1])>0.5).long()
        
        assert not (args.use_box_prompt and args.use_point_prompt)
        prompt_reflection = None
        if args.use_box_prompt:
            binary_cube_cropped = binary_cube[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection = (
                binary_cube_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0)
            )
        if args.use_point_prompt:
            binary_points_cropped = binary_points[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection = (
                binary_points_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0)
            )
    
        ## inference
        with torch.no_grad():
            logits_single_cropped = sliding_window_inference(
                    image_single_cropped.cuda(), prompt_reflection,
                    args.spatial_size, 1, segvol_model, args.infer_overlap,
                    text=text_single,
                    use_box=args.use_box_prompt,
                    use_point=args.use_point_prompt,
                )
            logits_single_cropped = logits_single_cropped.cpu().squeeze()
        logits_global_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = logits_single_cropped
        zoom_in_dice = dice_score(logits_global_single.squeeze(), label_single.squeeze())
        logits_labels_record[categories[item_idx]] = (
            zoom_in_dice,
            image_single, 
            points_single,
            box_single,
            logits_global_single, 
            label_single)
        print(f'===> zoom out dice {zoom_out_dice:.4f} -> zoom-out-zoom-in dice {zoom_in_dice:.4f} <===')
    return logits_labels_record

def inference_single_ct(args, segvol_model, data_item, categories):
    segvol_model.eval()
    image, gt3D = data_item["image"].float(), data_item["label"]
    image_zoom_out, gt3D__zoom_out = data_item["zoom_out_image"].float(), data_item['zoom_out_label']

    logits_labels_record = zoom_in_zoom_out(
        args, segvol_model, 
        image.unsqueeze(0), image_zoom_out.unsqueeze(0), 
        gt3D.unsqueeze(0), gt3D__zoom_out.unsqueeze(0),
        categories=categories)
    
    # visualize
    if args.visualize:
        for target, values in logits_labels_record.items():
            dice_score, image, point_prompt, box_prompt, logits, labels = values
            print(f'{target} result with Dice score {dice_score:.4f} visualizing')
            draw_result(target + f"-Dice {dice_score:.4f}", image, box_prompt, point_prompt, logits, labels, args.spatial_size, args.work_dir)


#######################################
args = set_parse()
## set path
model_names = ["SegVol_v1.pth",]
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
    # build model
    sam_model = sam_model_registry['vit'](args=args)

    segvol_model = SegVol(
                        image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        clip_ckpt=args.clip_ckpt,
                        roi_size=args.spatial_size,
                        patch_size=args.patch_size,
                        test_mode=args.test_mode,
                        ).cuda()


    ## load data
    all_test_transform = transforms.Compose([
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
        curGT_np = curGT.cpu().numpy()[0,...] # HWD
        # preGT_np = torch.permute(preGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
        ## convert to 3 channels
        # curMR = torch.cat((curMR, preMR, preGT), dim=1) # BCDHW
        img_np = curMR.cpu().numpy()[0, 0, ...] # HWD
        # img_np = (img_np * 255.0 - 123.67) / 58.39 ## ap
        curMR_np = torch.permute(curMR, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3

        ## inference
        binary_gt_orig_coords = binarize_gt_ap(curGT_np, affine_matrix)
        prompter_used = prmp.TenPoints3DVolumePrompter(
            model_infer,
            seed=2024
        )
        
        prompter_used.set_groundtruth(binary_gt_orig_coords)
        prompt_results = prompter_used.predict_image_ap(img_np, affine_matrix, SAMMed3DInferer)
        print(prompt_results.perf)

        pd = prompt_results.predicted_image.get_fdata()
        pd = torch.from_numpy(pd)[None, None, ...]
        # cal dice, hd95, nsd and asd
        # pd = torch.from_numpy(segs_3d_temp)[None, None, ...] #BCDHW
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
        pd_np = pd.cpu().numpy()[0, 0, ...].transpose((2, 0, 1)) # HWD-->DHW
        curGT_np = curGT_np.transpose((2, 0, 1)) 

        if args.visualize:
            seg_sitk = sitk.GetImageFromArray(pd_np)
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
