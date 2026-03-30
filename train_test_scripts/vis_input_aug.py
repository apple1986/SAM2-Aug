import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
import os
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


#####################################
def plot_overlay_boundary_mr(image_path, mask_path, output_path, slice_index=None, alpha=0.7):
    # Load image and mask
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()

    if image is None or mask is None:
        print(f"Error loading image or mask: {image_path}, {mask_path}")
        return

    # Determine the slice to display
    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Normalize image for display
    image_slice = (image_slice - np.min(image_slice)) / np.ptp(image_slice)
    image_rgb = np.stack([image_slice]*3, axis=-1)

    # Find mask boundary
    boundary = find_boundaries(mask_slice).astype(np.uint8)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb, cmap='gray')  # show full MR
    plt.contour(boundary, colors='yellow', linewidths=1.0)  # overlay boundary in red
    plt.axis('off')
    # plt.show()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Overlay image saved to {output_path}, with slice ID: {slice_index}")
    return slice_index


def plot_overlay_boundary_mr_crop(image_path, mask_path, output_path, slice_index=None, crop_coords=None, alpha=0.7):
    """
    Plot MR slice with overlayed mask boundary. Optionally crop image using mask or provided coordinates.

    Args:
        image_path (str): Path to MRI NIfTI image.
        mask_path (str): Path to mask NIfTI image.
        output_path (str): Path to save the output image.
        slice_index (int, optional): Slice index to visualize. If None, selects slice with max mask.
        crop_coords (tuple, optional): Tuple (y_min, y_max, x_min, x_max) to crop manually.
        alpha (float): Transparency for blending, if needed.
    """
    # Load image and mask
    image = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    # Select slice
    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Get cropping coordinates
    if crop_coords is None:
        nonzero_coords = np.argwhere((mask_slice > 0) | (image_slice > 0))
        y_min, x_min = nonzero_coords.min(axis=0)
        y_max, x_max = nonzero_coords.max(axis=0)
    else:
        y_min, y_max, x_min, x_max = crop_coords

    # Crop image and mask
    image_crop = image_slice[y_min:y_max+1, x_min:x_max+1]
    mask_crop = mask_slice[y_min:y_max+1, x_min:x_max+1]

    # Normalize MR image
    if np.ptp(image_crop) > 0:
        image_norm = (image_crop - np.min(image_crop)) / np.ptp(image_crop)
    else:
        image_norm = np.zeros_like(image_crop)

    image_rgb = np.stack([image_norm]*3, axis=-1)

    # Get boundary
    boundary = find_boundaries(mask_crop).astype(np.uint8)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb, cmap='gray')
    plt.contour(boundary, colors='yellow', linewidths=1.0)
    plt.axis('off')

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Overlay image saved to {output_path}, slice index: {slice_index}")
    return slice_index, [y_min, y_max, x_min, x_max]


def plot_mask(mask_path, output_path, slice_index=None, cmap='plasma'):
    # Load mask
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()

    # Determine the slice to display
    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    # Extract and normalize the mask slice
    mask_slice = mask[:, :, slice_index]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_slice, cmap=cmap)
    plt.axis('off')

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Mask image saved to {output_path}, slice index: {slice_index}")


def plot_mask_crop(mask_path, output_path, slice_index=None, crop_coords=None, cmap='plasma'):
    """
    Plot a mask slice, optionally cropping to specified coordinates.

    Args:
        mask_path (str): Path to the mask NIfTI file.
        output_path (str): Path to save the output image.
        slice_index (int, optional): Slice index to visualize. Auto-selected if None.
        crop_coords (tuple, optional): (y_min, y_max, x_min, x_max) to crop the mask.
        cmap (str): Colormap for mask visualization.
    """
    # Load mask
    mask = nib.load(mask_path).get_fdata()

    # Determine the slice to display
    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    mask_slice = mask[:, :, slice_index]

    # Crop if coordinates provided
    if crop_coords is not None:
        y_min, y_max, x_min, x_max = crop_coords
        mask_slice = mask_slice[y_min:y_max+1, x_min:x_max+1]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_slice, cmap=cmap)
    plt.axis('off')

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Mask image saved to {output_path}, slice index: {slice_index}")

#####################################


data_path = "/home/gxu/proj1/lesionSeg/dataset/HengRui_LiTS"
ID_ALL_PRE = ["70478710/70478710_20230928", "73121146/73121146_20231213", "73392758/73392758_20240521", "73777287/73777287_20240404", 
              "90058462/90058462_20240521", "90228632/90228632_20231117", "90472909/90472909_20231228", "90563174/90563174_20240510"]
ID_ALL_CUR = ["70478710/70478710_20231010", "73121146/73121146_20231229", "73392758/73392758_20240604", "73777287/73777287_20240418", 
              "90058462/90058462_20240611", "90228632/90228632_20231208", "90472909/90472909_20240116", "90563174/90563174_20240531"]


save_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/aug_input"
## load mr, overly mask, save png
## for the pre mr and gt
# for pre_id, cur_id in zip(ID_ALL_PRE, ID_ALL_CUR):
#     ## for previous
#     pre_mr_path = os.path.join(data_path, pre_id+"_mr.nii.gz")
#     pre_gt_path = os.path.join(data_path, pre_id+"_gt_gtv_combined.nii.gz")
#     subj_id = pre_mr_path.split('/')[-2]
#     save_name = os.path.join(save_path, subj_id+"_pre_im.png")
#     slice_id = plot_overlay_boundary_mr(pre_mr_path, pre_gt_path, save_name)
#     save_name = os.path.join(save_path, subj_id+"_pre_gt.png")
#     plot_mask(pre_gt_path, save_name, slice_id, cmap="gray")

#     ## for current
#     cur_mr_path = os.path.join(data_path, cur_id+"_mr.nii.gz")
#     cur_gt_path = os.path.join(data_path, cur_id+"_gt_gtv_combined.nii.gz")
#     save_name = os.path.join(save_path, subj_id+"_cur_im.png")
#     plot_overlay_boundary_mr(cur_mr_path, cur_gt_path, save_name, slice_id)
#     save_name = os.path.join(save_path, subj_id+"_cur_gt.png")
#     plot_mask(cur_gt_path, save_name, slice_id, cmap="gray")



## crop the mri
for pre_id, cur_id in zip(ID_ALL_PRE, ID_ALL_CUR):
    ## for previous
    pre_mr_path = os.path.join(data_path, pre_id+"_mr.nii.gz")
    pre_gt_path = os.path.join(data_path, pre_id+"_gt_gtv_combined.nii.gz")
    subj_id = pre_mr_path.split('/')[-2]
    save_name = os.path.join(save_path, subj_id+"_pre_im_crop.png")
    slice_id,  crop_coords = plot_overlay_boundary_mr_crop(pre_mr_path, pre_gt_path, save_name)
    save_name = os.path.join(save_path, subj_id+"_pre_gt_crop.png")
    plot_mask_crop(pre_gt_path, save_name, slice_id, crop_coords, cmap="gray")

    ## for current
    cur_mr_path = os.path.join(data_path, cur_id+"_mr.nii.gz")
    cur_gt_path = os.path.join(data_path, cur_id+"_gt_gtv_combined.nii.gz")
    save_name = os.path.join(save_path, subj_id+"_cur_im_crop.png")
    plot_overlay_boundary_mr_crop(cur_mr_path, cur_gt_path, save_name, slice_id, crop_coords)
    save_name = os.path.join(save_path, subj_id+"_cur_gt_crop.png")
    plot_mask_crop(cur_gt_path, save_name, slice_id, crop_coords, cmap="gray")