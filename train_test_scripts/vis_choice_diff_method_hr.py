from pyexpat import model
import sys

from torch import gt

sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
import os
from gp_knife.plot_overlay_nii_np import overlay_pd_gt_onMR
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path as osp
from itertools import product
from gp_knife.paper_fig_plot import plot_overlay_bbox_all, crop_black_background, plot_overlay_mask_boundary_all, crop_black_background_basedMR
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage import measure
import torch
import random
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from matplotlib.patches import Rectangle
from glob import glob
###########################################################

def plot_mr_with_bound_and_bbox(image_slice, mask_slice, output_path=None, bbox_size=(50, 50)):
    """
    Plot MR slice with tumor boundary (red) and bounding box (yellow) of specified size.
    """
    image_rgb = np.stack([image_slice]*3, axis=-1)

    # Step 1: Extract boundary from mask
    boundary = find_boundaries(mask_slice, mode='inner').astype(bool)

    # Step 2: Tumor region center for bbox
    regions = regionprops(mask_slice)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb, cmap='gray')

    # Step 3: Plot boundary as red edges
    if np.any(boundary):
        ys, xs = np.nonzero(boundary)
        plt.plot(xs, ys, 'r.', markersize=1.5)
    else:
        print("Warning: No boundary found in mask_slice")

    # Step 4: Draw bounding box around each tumor
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        center_r = (minr + maxr) // 2
        center_c = (minc + maxc) // 2
        height, width = bbox_size

        bbox_minr = max(0, center_r - height // 2)
        bbox_maxr = min(image_slice.shape[0], center_r + height // 2)
        bbox_minc = max(0, center_c - width // 2)
        bbox_maxc = min(image_slice.shape[1], center_c + width // 2)

        plt.gca().add_patch(plt.Rectangle((bbox_minc, bbox_minr),
                                          bbox_maxc - bbox_minc,
                                          bbox_maxr - bbox_minr,
                                          edgecolor='green', linewidth=2, linestyle='--', facecolor='none'))

    plt.axis('off')

    # Step 5: Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()
## with crop black background
def crop_nonzero_area(image, mask):
    """Crop both image and mask to non-zero region in the image."""
    rows = np.any(image > 10, axis=1)
    cols = np.any(image > 10, axis=0)

    if not rows.any() or not cols.any():
        return image, mask  # No non-zero region found

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add a small margin (optional)
    margin = 5
    rmin = max(0, rmin - margin)
    rmax = min(image.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(image.shape[1], cmax + margin)

    return image[rmin:rmax+1, cmin:cmax+1], mask[rmin:rmax+1, cmin:cmax+1]

def plot_mr_with_bound_and_bbox_crop(image_slice, mask_slice, output_path=None, bbox_size=(50, 50)):
    """
    Plot cropped MR slice with tumor boundary and green dashed bounding box.
    """
    # Step 1: Crop image and mask
    # image_crop, mask_crop = crop_nonzero_area(image_slice, mask_slice)
    image_crop, mask_crop = image_slice, mask_slice

    # Step 2: Prep RGB
    image_rgb = np.stack([image_crop]*3, axis=-1)

    # Step 3: Mask boundary
    boundary = find_boundaries(mask_crop, mode='inner').astype(bool)
    regions = regionprops(mask_crop)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb, cmap='gray')

    # Step 4: Red boundary dots
    if np.any(boundary):
        ys, xs = np.nonzero(boundary)
        plt.plot(xs, ys, 'r.', markersize=1.5)
    else:
        print("Warning: No boundary found in mask_slice")

    # Step 5: Green dashed bbox
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        center_r = (minr + maxr) // 2
        center_c = (minc + maxc) // 2
        height, width = bbox_size

        bbox_minr = max(0, center_r - height // 2)
        bbox_maxr = min(image_crop.shape[0], center_r + height // 2)
        bbox_minc = max(0, center_c - width // 2)
        bbox_maxc = min(image_crop.shape[1], center_c + width // 2)

        plt.gca().add_patch(plt.Rectangle(
            (bbox_minc, bbox_minr),
            bbox_maxc - bbox_minc,
            bbox_maxr - bbox_minr,
            edgecolor='yellow',
            linestyle='--',
            linewidth=2,
            facecolor='none'
        ))

    plt.axis('off')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


## extract bbox of the tumor region
def plot_cropped_bbox_with_mask(image_slice, mask_slice, output_dir=None, alpha=0.5, bbox_size=(50, 50)):
    """
    For each object in the mask, extract its bounding box, crop the MR and mask,
    and overlay the mask (in red) on the MR in that cropped region.
    """
    import os
    from skimage.measure import regionprops
    import matplotlib.pyplot as plt
    import numpy as np

    regions = regionprops(mask_slice)

    for idx, region in enumerate(regions):
        # Compute center and fixed-size bbox
        minr, minc, maxr, maxc = region.bbox
        center_r = (minr + maxr) // 2
        center_c = (minc + maxc) // 2
        height, width = bbox_size

        bbox_minr = max(0, center_r - height // 2)
        bbox_maxr = min(image_slice.shape[0], center_r + height // 2)
        bbox_minc = max(0, center_c - width // 2)
        bbox_maxc = min(image_slice.shape[1], center_c + width // 2)

        # Crop MR and mask
        cropped_mr = image_slice[bbox_minr:bbox_maxr, bbox_minc:bbox_maxc]
        cropped_mask = mask_slice[bbox_minr:bbox_maxr, bbox_minc:bbox_maxc]

        # Make RGB MR image
        cropped_rgb = np.stack([cropped_mr]*3, axis=-1)

        # Plot
        plt.figure(figsize=(5, 5))
        plt.imshow(cropped_rgb, cmap='gray')

        # Create red overlay mask (RGBA)
        red_overlay = np.zeros((*cropped_mask.shape, 4), dtype=np.float32)
        red_overlay[..., 0] = 1.0  # Red
        red_overlay[..., 3] = (cropped_mask > 0) * alpha  # Alpha mask
        plt.imshow(red_overlay)

        plt.axis('off')

        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"crop_{idx:02d}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"Saved cropped overlay to {out_path}")
        else:
            plt.title(f"Crop {idx}")
            plt.show()

        plt.close()

## overlay pd on MR with gt boundary
def plot_cropped_bbox_with_mask_boundary(image_slice, mask_slice, output_dir=None, bbox_size=(50, 50)):
    """
    Crop the MR and overlay the exact red boundary (pixel aligned) of the mask.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries

    regions = regionprops(mask_slice)

    for idx, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        center_r = (minr + maxr) // 2
        center_c = (minc + maxc) // 2
        height, width = bbox_size

        bbox_minr = max(0, center_r - height // 2)
        bbox_maxr = min(image_slice.shape[0], center_r + height // 2)
        bbox_minc = max(0, center_c - width // 2)
        bbox_maxc = min(image_slice.shape[1], center_c + width // 2)

        cropped_mr = image_slice[bbox_minr:bbox_maxr, bbox_minc:bbox_maxc]
        cropped_mask = mask_slice[bbox_minr:bbox_maxr, bbox_minc:bbox_maxc]

        boundary = find_boundaries(cropped_mask, mode='outer')

        cropped_rgb = np.stack([cropped_mr]*3, axis=-1)

        plt.figure(figsize=(5, 5))
        plt.imshow(cropped_rgb, cmap='gray')

        # Pixel-aligned red boundary
        boundary_rgba = np.zeros((*boundary.shape, 4), dtype=np.float32)
        boundary_rgba[..., 0] = 1.0  # Red
        boundary_rgba[..., 3] = boundary.astype(np.float32)  # Alpha mask
        plt.imshow(boundary_rgba)

        plt.axis('off')

        if output_dir:
            # os.makedirs(output_dir, exist_ok=True)
            # out_path = os.path.join(output_dir, f"crop_boundary_{idx:02d}.png")
            plt.savefig(output_dir, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"Saved cropped boundary overlay to {output_dir}")
        else:
            plt.title(f"Boundary Crop {idx}")
            plt.show()

        plt.close()

def crop_and_overlay_mr_gt_pred(image_slice, mask_gt, mask_pred, bbox_size=(50, 50), output_path=None):
    """
    Crop the MR image around the ground truth mask's bounding box,
    overlay GT boundary (red) and predicted mask (green fill) on the cropped MR.
    """
    # Ensure binary masks
    mask_gt = (mask_gt > 0).astype(np.uint8)
    mask_pred = (mask_pred > 0).astype(np.uint8)

    # Get bounding box from ground truth mask
    props = regionprops(mask_gt)
    if not props:
        print("No object found in GT mask.")
        return

    minr, minc, maxr, maxc = props[0].bbox
    center_r = (minr + maxr) // 2
    center_c = (minc + maxc) // 2
    height, width = bbox_size

    # Compute bounding box with fixed size (bbox_size)
    bbox_minr = max(0, center_r - height // 2)
    bbox_maxr = min(image_slice.shape[0], center_r + height // 2)
    bbox_minc = max(0, center_c - width // 2)
    bbox_maxc = min(image_slice.shape[1], center_c + width // 2)

    # Crop the MR, GT mask, and Pred mask based on computed bbox
    cropped_mr = image_slice[bbox_minr:bbox_maxr, bbox_minc:bbox_maxc]
    cropped_gt = mask_gt[bbox_minr:bbox_maxr, bbox_minc:bbox_maxc]
    cropped_pred = mask_pred[bbox_minr:bbox_maxr, bbox_minc:bbox_maxc]

    # Find GT boundary
    boundary_gt = find_boundaries(cropped_gt, mode='outer')

    # Create RGB image for MR (scaled to 0-1 for better visualization)
    mr_rgb = np.stack([cropped_mr] * 3, axis=-1)
    mr_rgb = mr_rgb.astype(np.float32) / (np.max(mr_rgb) + 1e-5)

    # Plot the MR with overlays
    plt.figure(figsize=(6, 6))
    plt.imshow(mr_rgb, cmap='gray')

    # Overlay predicted mask (green with transparency)
    green_overlay = np.zeros((*cropped_pred.shape, 4), dtype=np.float32)
    green_overlay[..., 1] = 1.0  # Green channel
    green_overlay[..., 3] = 0.4 * cropped_pred  # Alpha channel
    plt.imshow(green_overlay)

    # Overlay GT boundary (red)
    red_boundary = np.zeros((*boundary_gt.shape, 4), dtype=np.float32)
    red_boundary[..., 0] = 1.0  # Red channel
    red_boundary[..., 3] = boundary_gt.astype(np.float32)
    plt.imshow(red_boundary)

    plt.axis('off')

    # Save or show the result
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved cropped overlay to {output_path}")
    else:
        plt.show()

    plt.close()
################################################################
root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir"
save_fig_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/compare_diff_method_hr"
root_cnn_tras_path = "/home/gxu/proj1/lesionSeg/utswlits3d/checkpoint"

sam2_aug_path = osp.join(root_path, "paper_used_model/Sam2_Tiny_Promp_Aug_3Ch/AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5/dataset_hr/seg_result_itv")
sam2_ori_path = osp.join(root_path, "paper_used_model/Sam2_Tiny_Promp_Aug_3Ch/AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5/dataset_hr/sam2_ori_seg_result_itv")
sammed2d_path = osp.join(root_path, "paper_used_model/Sam2_Tiny_Promp_Aug_3Ch/AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5/dataset_hr/sam-med2d_b_seg_result_itv")
medsam_path = osp.join(root_path, "paper_used_model/Sam2_Tiny_Promp_Aug_3Ch/AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5/dataset_hr/medsam_seg_result_itv")
mednext_path = osp.join(root_cnn_tras_path, "mednextv1_medium/mednextv1_medium_DiceCELoss_3ch_2bs_64sz_case29_gp/result_hr")
segresnet_path = osp.join(root_cnn_tras_path, "SegResNet/SegResNet_DiceCELoss_3ch_2bs_64sz_case29_gp/result_hr")
swinunet_path = osp.join(root_cnn_tras_path, "SwinUNETR/SwinUNETR_DiceCELoss_3ch_2bs_64sz_case29_gp/result_hr")
segformer_path = osp.join(root_cnn_tras_path, "SegFormer3D/SegFormer3D_DiceCELoss_3ch_2bs_64sz_case29_gp/result_hr")

## methods and cases
# method_paths = [sam2_aug_path, sam2_ori_path, sammed2d_path, medsam_path, mednext_path, segresnet_path, swinunet_path, segformer_path]
# MODEL_NAMES = ["sam2_aug", "sam2_ori", "sammed2d", "medsam", "mednext", "segresnet", "swinunet", "segformer"]

method_paths = [sam2_aug_path, sammed2d_path, medsam_path,  segresnet_path, segformer_path,]
MODEL_NAMES = ["sam2_aug", "sammed2d", "medsam",  "segresnet", "segformer_path",]
case_names = ["70995377_20240119", "72733568_20240126", 
                "73121146_20231229", "73392758_20240604",
                "90033790_20240326", "90058462_20240611","90228632_20231208",
                "90563174_20240531", "90936003_20240627", "90942785_20240611","91589196_20240523",
                "92183907_20240605", "92229019_20240122", "92787492_20240513",
                "93551831_20240305", "94459035_20230705", "94530693_20230830", "95716607_20240208",
                "95722247_20231206", "95828813_20240429", "96039906_20240703", "96710759_20231201",
                "96861591_20240108", "97124281_20240604", "97256660_20230901", "97790476_20240626",
                "97792439_20231214", "97947209_20240403"
                
                ] # 

# select slice idx that has the max dice for sam2-aug
sel_idxs = []
for case_name in case_names:
    # open GT, PD
    gt_path = os.path.join(method_paths[0], f"{case_name}_gt_sam2.nii.gz")
    pd_path = os.path.join(method_paths[0], f"{case_name}_pd_sam2.nii.gz")
    gt_vol_obj = nib.load(gt_path)
    gt_vol = gt_vol_obj.get_fdata()
    pd_vol_obj = nib.load(pd_path)
    pd_vol = pd_vol_obj.get_fdata()
    # calculate 3D dice
    dice_vol = 2 * np.sum((gt_vol * pd_vol) > 0) / (np.sum(gt_vol > 0) + np.sum(pd_vol > 0)+1e-5)
    print(f"3D dice: {dice_vol}")

    # find the slice with the max dice
    max_dice = 0
    for i in range(gt_vol.shape[2]):
        gt_slice = gt_vol[:, :, i]
        pd_slice = pd_vol[:, :, i]
        # calculate dice
        dice = 2* np.sum((gt_slice * pd_slice) > 0) / (np.sum(gt_slice > 0) + np.sum(pd_slice > 0)+1e-5)
        if dice > max_dice:
            max_dice = dice
            sel_idx = i
    sel_idxs.append(sel_idx)
print(sel_idxs)

def cal_dice(gt_slice, pd_slice):
    # calculate dice
    dice = 2* np.sum((gt_slice * pd_slice) > 0) / (np.sum(gt_slice > 0) + np.sum(pd_slice > 0)+1e-5)
    return dice 

## set hyper-parameters
BBOX_SIZE = 96
# select one method
for method_path, model_name in zip(method_paths, MODEL_NAMES):
    print(f"method: {model_name}")
    # select one case
    for case_name, idx in zip(case_names, sel_idxs):
        # set path
        mr_path = os.path.join(method_paths[0], f"{case_name}_im_sam2.nii.gz")
        # gt_path = os.path.join(method_path, f"{case_name}_gt_sam2.nii.gz")
        # pd_path = os.path.join(method_path, f"{case_name}_pd_sam2.nii.gz")
        gt_path = glob(os.path.join(method_path, f"{case_name}_*gt*.nii.gz"))[0]
        pd_path = glob(os.path.join(method_path, f"{case_name}_*pd*.nii.gz"))[0]

        # open MR, GT, PD
        mr_vol_obj = nib.load(mr_path)
        gt_vol_obj = nib.load(gt_path)
        pd_vol_obj = nib.load(pd_path)
        mr_vol = mr_vol_obj.get_fdata()
        gt_vol = gt_vol_obj.get_fdata()
        pd_vol = pd_vol_obj.get_fdata()

        # select slice idx that has the max dice
        # select slice idx that between the idx-5 and idx+5
        for idx_sel in range(idx-0, idx+1):
            mr_slice = mr_vol[:, :, idx_sel]
            mr_slice = (mr_slice - mr_slice.min()) / (mr_slice.max() - mr_slice.min()) * 255
            mr_slice = mr_slice.astype(np.uint8)
            gt_slice = gt_vol[:, :, idx_sel].astype(np.uint8)
            gt_slice = (gt_slice > 0).astype(np.uint8)
            pd_slice = pd_vol[:, :, idx_sel].astype(np.uint8)
            pd_slice = (pd_slice > 0).astype(np.uint8)

            # 1. overlay ground truth boundary and bounding box on MR
            if "sam2_aug" in model_name:
                save_name = osp.join(save_fig_path, f"{case_name}_s{idx_sel}_{model_name}_gt_bound_bbox_on_mr.png")
                plot_mr_with_bound_and_bbox_crop(mr_slice, gt_slice, output_path=save_name, bbox_size=(BBOX_SIZE, BBOX_SIZE))
                save_name = osp.join(save_fig_path, f"{case_name}_s{idx_sel}_{model_name}_gt_bound_on_mr_crop.png")
                plot_cropped_bbox_with_mask_boundary(mr_slice, gt_slice, output_dir=save_name, bbox_size=(BBOX_SIZE, BBOX_SIZE))
                ## cal dice
                dice_slice = cal_dice(gt_slice, pd_slice)
                save_name = osp.join(save_fig_path, f"{case_name}_s{idx_sel}_{model_name}_gt_bound_pd_mask_on_mr_crop_dice{dice_slice*100:.2f}.png")
                crop_and_overlay_mr_gt_pred(mr_slice, gt_slice, pd_slice, bbox_size=(BBOX_SIZE, BBOX_SIZE), output_path=save_name)
            else:
                dice_slice = cal_dice(gt_slice, pd_slice)
                save_name = osp.join(save_fig_path, f"{case_name}_s{idx_sel}_{model_name}_gt_bound_pd_mask_on_mr_crop_dice{dice_slice*100:.2f}.png")
                crop_and_overlay_mr_gt_pred(mr_slice, gt_slice, pd_slice, bbox_size=(BBOX_SIZE, BBOX_SIZE), output_path=save_name)