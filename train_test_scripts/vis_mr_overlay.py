import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
import os
from gp_knife.paper_fig_plot import plot_overlay_bbox_all, crop_black_background, plot_overlay_mask_boundary_all, crop_black_background_basedMR
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt


################################
def plot_overlay_mask_all(image_path, mask_path, output_path, slice_index=None):
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()
    
    if image is None or mask is None:
        print(f"Error loading image or mask: {image_path}, {mask_path}")
        return
    
    # Assuming the image and mask are 3D and we want to plot the slice with the most mask content
    slice_index = np.argmax(np.sum(mask, axis=(0, 1)))
    image_slice = image[:, :, slice_index]
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
    image_slice = image_slice.astype(np.uint8)
    mask_slice = mask[:, :, slice_index].astype(np.uint8)
    
    # Convert grayscale MRI image to BGR for color overlay
    mri_colored = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2RGB)

    # Create an RGBA image for transparent overlay
    overlay = cv2.cvtColor(mri_colored, cv2.COLOR_RGB2RGBA)

    # Set the alpha channel to be fully transparent (0) where the mask is 0
    alpha = np.zeros_like(mask_slice, dtype=np.uint8)
    alpha[mask_slice > 0] = 30  # Adjust transparency level (0-255)

    # Apply the mask color (red) only on non-zero mask regions
    red_overlay = np.zeros_like(overlay)
    red_overlay[..., 0] = 125  # Red channel
    red_overlay[..., 3] = alpha  # Alpha channel

    # Blend the red mask onto the MRI image
    combined = overlay.copy()
    mask_region = (red_overlay[..., 3] > 0)
    combined[mask_region] = red_overlay[mask_region]  # Apply red color with transparency

    # Display the image with mask overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_RGBA2RGB))  # Convert RGBA to RGB for display
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Overlay image saved to {output_path}")

def plot_overlay_erode_mask_all(image_path, mask_path, output_path, slice_index=None):
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()
    
    if image is None or mask is None:
        print(f"Error loading image or mask: {image_path}, {mask_path}")
        return
    
    # Assuming the image and mask are 3D and we want to plot the slice with the most mask content
    slice_index = np.argmax(np.sum(mask, axis=(0, 1)))
    image_slice = image[:, :, slice_index]
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
    image_slice = image_slice.astype(np.uint8)
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Apply erosion to the mask
    kernel = np.ones((11,11), np.uint8)  # Define kernel size for erosion
    mask_slice = cv2.erode(mask_slice, kernel, iterations=1)
    
    # Convert grayscale MRI image to BGR for color overlay
    mri_colored = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2RGB)

    # Create an RGBA image for transparent overlay
    overlay = cv2.cvtColor(mri_colored, cv2.COLOR_RGB2RGBA)

    # Set the alpha channel to be fully transparent (0) where the mask is 0
    alpha = np.zeros_like(mask_slice, dtype=np.uint8)
    alpha[mask_slice > 0] = 30  # Adjust transparency level (0-255)

    # Apply the mask color (red) only on non-zero mask regions
    red_overlay = np.zeros_like(overlay)
    red_overlay[..., 1] = 125  # Red channel
    red_overlay[..., 3] = alpha  # Alpha channel

    # Blend the red mask onto the MRI image
    combined = overlay.copy()
    mask_region = (red_overlay[..., 3] > 0)
    combined[mask_region] = red_overlay[mask_region]  # Apply red color with transparency

    # Display the image with mask overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_RGBA2RGB))  # Convert RGBA to RGB for display
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Overlay image saved to {output_path}")



def save_mr_according_to_mask(image_path, mask_path, output_path, slice_index=None):
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()
    
    if image is None or mask is None:
        print(f"Error loading image or mask: {image_path}, {mask_path}")
        return
    
    # If slice_index is not given, select the slice with the most mask content
    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))
        slice_index = slice_index
    
    image_slice = image[:, :, slice_index]
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
    image_slice = image_slice.astype(np.uint8)
    # mask_slice = mask[:, :, slice_index].astype(np.uint8)
    
    # # Mask the image by multiplying with the binary mask
    # masked_image = cv2.bitwise_and(image_slice, image_slice, mask=mask_slice)
    
    # Save the masked image without cropping
    plt.figure(figsize=(8, 8))
    plt.imshow(image_slice, cmap="gray")
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Masked MRI slice saved to {output_path}")

def save_mask(mask_path, output_path, slice_index=None):
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()

    
    # If slice_index is not given, select the slice with the most mask content
    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))
        slice_index = slice_index
    

    mask_slice = mask[:, :, slice_index].astype(np.uint8)
    
    # # Mask the image by multiplying with the binary mask
    # masked_image = cv2.bitwise_and(image_slice, image_slice, mask=mask_slice)
    
    # Save the masked image without cropping
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_slice, cmap="gray")
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Masked slice saved to {output_path}")

################################
"""
sample ID
70461960_20241114_im_sam2.nii.gz
72056839_20241008_im_sam2.nii.gz
73432449_20240809_im_sam2.nii.gz
73660494_20240826_im_sam2.nii.gz
"""
root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2"
data_path = os.path.join(root_path, "sam_methods/MedSAM2/work_dir/paper_used_model/Sam2_Tiny_Promp_Aug_3Ch/FT_Img_Msk_Prmp_3ch_Rand5/seg_result")
mr_paths = [
            os.path.join(data_path, "70461960_20241114_im_sam2.nii.gz"), os.path.join(data_path, "72056839_20241008_im_sam2.nii.gz"),
            os.path.join(data_path, "73432449_20240809_im_sam2.nii.gz"), os.path.join(data_path, "73660494_20240826_im_sam2.nii.gz"),
            os.path.join(data_path, "70125695_20240828_im_sam2.nii.gz"), os.path.join(data_path, "73401174_20240828_im_sam2.nii.gz"),
            ] 
gt_paths = [
            os.path.join(data_path, "70461960_20241114_gt_sam2.nii.gz"), os.path.join(data_path, "72056839_20241008_gt_sam2.nii.gz"),
            os.path.join(data_path, "73432449_20240809_gt_sam2.nii.gz"), os.path.join(data_path, "73660494_20240826_gt_sam2.nii.gz"),
            os.path.join(data_path, "70125695_20240828_gt_sam2.nii.gz"), os.path.join(data_path, "73401174_20240828_gt_sam2.nii.gz"),
            ]
save_fig_path = os.path.join(root_path, "paper_vis/bbox_bound_overlay_MR")

NUM_CASE = 0
for NUM_CASE in range(4, 6): # len(mr_paths)
    # IDX_SLICE = 54
    # 1. open the files
    mr_vol_obj = nib.load(mr_paths[NUM_CASE])
    gt_vol_obj = nib.load(gt_paths[NUM_CASE])
    mr_vol = mr_vol_obj.get_fdata()
    gt_vol = gt_vol_obj.get_fdata()
    print(gt_vol.shape)
    case_id_tmp = os.path.basename(mr_paths[NUM_CASE]).split("_")
    case_id = case_id_tmp[0] + "_" + case_id_tmp[1]

    # 2. bbox overlay mr, expand bbox to 5 pixels
    save_name = "fig1_bboxMR_"+case_id+".png"
    output_path = os.path.join(save_fig_path, save_name)
    plot_overlay_bbox_all(mr_paths[NUM_CASE], gt_paths[NUM_CASE], output_path, slice_index=None)
    # post-process: crop black background
    img_path = output_path
    save_name = "fig1_bboxMR_"+case_id+"_crop.png"
    save_path = os.path.join(save_fig_path, save_name)
    crop_black_background(img_path, save_path)

    # 3. gt bpundary overlay mr
    save_name = "fig1_boundMR_"+case_id+".png"
    output_path = os.path.join(save_fig_path, save_name)
    plot_overlay_mask_boundary_all(mr_paths[NUM_CASE], gt_paths[NUM_CASE], output_path, slice_index=None)
    # post-process: crop black background
    img_path = output_path
    save_name = "fig1_boundMR_"+case_id+"_crop.png"
    save_path = os.path.join(save_fig_path, save_name)
    crop_black_background(img_path, save_path)

    # # 4. save mask overlay mr
    # save_name = "fig1_maskMR_"+case_id+".png"
    # output_path = os.path.join(save_fig_path, save_name)
    # plot_overlay_mask_all(mr_paths[NUM_CASE], gt_paths[NUM_CASE], output_path, slice_index=None)
    # # post-process: crop black background
    # img_path = output_path
    # save_name = "fig1_maskMR_"+case_id+"_crop.png"
    # save_path = os.path.join(save_fig_path, save_name)
    # crop_black_background(img_path, save_path)

    # # 5. save original mr
    # save_name = "fig1_oriMR_"+case_id+".png"
    # output_path = os.path.join(save_fig_path, save_name)
    # save_mr_according_to_mask(mr_paths[NUM_CASE], gt_paths[NUM_CASE],output_path)
    # # post-process: crop black background
    # img_path = output_path
    # save_name = "fig1_oriMR_"+case_id+"_crop.png"
    # save_path = os.path.join(save_fig_path, save_name)
    # crop_black_background(img_path, save_path)

    # # 6. ovely erode mask on MR
    # save_name = "fig1_erodeMR_"+case_id+".png"
    # output_path = os.path.join(save_fig_path, save_name)
    # plot_overlay_erode_mask_all(mr_paths[NUM_CASE], gt_paths[NUM_CASE], output_path, slice_index=None)
    # # post-process: crop black background
    # img_path = output_path
    # save_name = "fig1_erodeMR_"+case_id+"_crop.png"
    # save_path = os.path.join(save_fig_path, save_name)
    # crop_black_background(img_path, save_path)

    # # 7. save mask
    # save_name = "fig1_mask_"+case_id+"_move5s.png"
    # output_path = os.path.join(save_fig_path, save_name)
    # save_mask(gt_paths[NUM_CASE], output_path, slice_index=None)
    # # post-process: crop black background
    # img_path = output_path
    # mr_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/fig1_oriMR_72056839_20241008_move5s.png"
    # save_name = "fig1_mask_"+case_id+"_move5s_crop.png"
    # save_path = os.path.join(save_fig_path, save_name)
    # crop_black_background_basedMR(mr_path, img_path, save_path)