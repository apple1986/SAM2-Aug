import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
import os
from gp_knife.plot_overlay_nii_np import overlay_pd_gt_onMR
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
m1: model 1: 1ch-2ft-s5-mdc-0.7744: input: curMR, ft: imageEnc and maskDec, mean dice: 0.7744
m2: model 2: 3ch-3ft-s5-mdc-0.8551: input: curMR,preMR, preGT ft: imageEnc, maskDec and promptEnc, mean dice: 0.8551
1. GT boundary overlay MR
2. pd mask overly GT boundary overlay MR
3. bbox enlarge-5 overlay MR
"""

root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir"
m1_case_path_mr = os.path.join(root_path, "Sam2_Tiny_BBox_Shift/MedSAM2-UTSW-GP-FT-ALL-Shift_Rand5/seg_result/70125695_20240828_im_sam2.nii.gz")
m1_case_path_gt = os.path.join(root_path, "Sam2_Tiny_BBox_Shift/MedSAM2-UTSW-GP-FT-ALL-Shift_Rand5/seg_result/70125695_20240828_gt_sam2.nii.gz")
m1_case_path_pd = os.path.join(root_path, "Sam2_Tiny_BBox_Shift/MedSAM2-UTSW-GP-FT-ALL-Shift_Rand5/seg_result/70125695_20240828_pd_sam2.nii.gz")
m2_case_path_mr = os.path.join(root_path, "Sam2_Tiny_Promp_Aug_3Ch/FT_Img_Msk_Prmp_3ch_Rand5/seg_result/70125695_20240828_im_sam2.nii.gz")
m2_case_path_gt = os.path.join(root_path, "Sam2_Tiny_Promp_Aug_3Ch/FT_Img_Msk_Prmp_3ch_Rand5/seg_result/70125695_20240828_gt_sam2.nii.gz")
m2_case_path_pd = os.path.join(root_path, "Sam2_Tiny_Promp_Aug_3Ch/FT_Img_Msk_Prmp_3ch_Rand5/seg_result/70125695_20240828_pd_sam2.nii.gz")
save_fig_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis"

# open mr, gt, pd
mr_vol_obj = nib.load(m1_case_path_mr)
gt_vol_obj = nib.load(m1_case_path_gt)
pd_vol_obj = nib.load(m1_case_path_pd)
mr_vol = mr_vol_obj.get_fdata()
gt_vol = gt_vol_obj.get_fdata()
pd_vol = pd_vol_obj.get_fdata()

# select slice idx
IDX = 35
# 1. overlay boundary on MR
mr_slice = mr_vol[:, :, IDX]
mr_slice = (mr_slice - mr_slice.min()) / (mr_slice.max() - mr_slice.min()) * 255
mr_slice = mr_slice.astype(np.uint8)
gt_slice = gt_vol[:, :, IDX].astype(np.uint8)
pd_slice = pd_vol[:, :, IDX].astype(np.uint8)
print(f"max val: {mr_slice.max()}")
# # Find contours from the binary mask
# contours, _ = cv2.findContours(gt_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # Convert grayscale MRI image to BGR for color overlay
# mri_colored = cv2.cvtColor(mr_slice, cv2.COLOR_GRAY2BGR)
# # Draw the contours (boundary) on the image in red (or any color)
# cv2.drawContours(mri_colored, contours, -1, (0, 0, 255), thickness=1)  # Red boundary
# # Display the image with boundary overlay
# plt.figure(figsize=(8, 8))
# plt.imshow(cv2.cvtColor(mri_colored, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
# plt.axis("off")
# # plt.title("MRI Image with Boundary Overlay")
# # plt.show()
# plt.savefig(os.path.join(save_fig_path, "70125695_20240828_s35_gt_onMR.png"), dpi=300, bbox_inches='tight')

# 2. pd overlay MR with gt boundary
# save_name_m2 = os.path.join(save_fig_path, "70125695_20240828_s35_sam2_3ch_3ft_aug.png")
save_name_m1 = os.path.join(save_fig_path, "70125695_20240828_s35_sam2_1ch_2ft_aug.png")
overlay_pd_gt_onMR(mr_slice, pd_slice, gt_slice, alpha=0.5, save_name=save_name_m1)


