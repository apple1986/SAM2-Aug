import sys

from matplotlib import lines
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

##########################################
def plot_overlay_morph_mask_all(image_path, mask_path, output_path, slice_index=None, 
                                 operation='erode', kernel_size=11, iterations=1, alpha=0.7):
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

    # Process the mask: erode or dilate
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'erode':
        processed_mask = cv2.erode(mask_slice, kernel, iterations=iterations)
    elif operation == 'dilate':
        processed_mask = cv2.dilate(mask_slice, kernel, iterations=iterations)
    else:
        raise ValueError("Invalid operation. Use 'erode' or 'dilate'.")

    # Create original mask boundary
    original_boundary = find_boundaries(mask_slice).astype(np.uint8)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb, cmap='gray')
    
    # Show processed mask (filled) — use a custom overlay color
    plt.imshow(np.ma.masked_where(processed_mask == 0, processed_mask), 
               cmap=plt.colormaps.get_cmap('spring'), alpha=alpha) # 'magenta', 'spring', 'cool', 'copper', 'viridis', 'plasma', 'Oranges', etc. summer
    
    # Show original mask boundary in red
    plt.contour(original_boundary, colors='red', linewidths=1.0)

    plt.title(f"Slice {slice_index} – {operation.capitalize()} Mask Overlay", fontsize=14)
    plt.axis('off')

    # Save or show
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f"Overlay image saved to {output_path}")

#######################################
## crop according to the ROI
def extract_roi(image_slice, mask_slice, size=64):
    ys, xs = np.where(mask_slice > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No mask content found to center the ROI.")

    x_center = int(np.mean(xs))
    y_center = int(np.mean(ys))

    half_size = size // 2
    x_start = max(x_center - half_size, 0)
    y_start = max(y_center - half_size, 0)
    x_end = x_start + size
    y_end = y_start + size

    # Keep ROI within image bounds
    x_end = min(x_end, image_slice.shape[1])
    y_end = min(y_end, image_slice.shape[0])
    x_start = x_end - size
    y_start = y_end - size

    image_roi = image_slice[y_start:y_end, x_start:x_end]
    mask_roi = mask_slice[y_start:y_end, x_start:x_end]
    
    return image_roi, mask_roi

def plot_overlay_roi_mask(image_path, mask_path, output_path, slice_index=None,
                          operation='erode', kernel_size=11, iterations=1,
                          overlay_cmap='hot', alpha=0.5, boundary_color='red',
                          roi_size=128):
    # Load NIfTI
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()

    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Normalize MRI
    image_slice = (image_slice - np.min(image_slice)) / np.ptp(image_slice)

    # Apply morphological operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'erode':
        processed_mask = cv2.erode(mask_slice, kernel, iterations=iterations)
    elif operation == 'dilate':
        processed_mask = cv2.dilate(mask_slice, kernel, iterations=iterations)
    else:
        raise ValueError("Invalid operation. Use 'erode' or 'dilate'.")

    # Crop 64x64 ROI from both image and processed mask
    image_roi, processed_mask_roi = extract_roi(image_slice, processed_mask, size=roi_size)
    _, original_mask_roi = extract_roi(image_slice, mask_slice, size=roi_size)
    boundary_roi = find_boundaries(original_mask_roi).astype(np.uint8)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(image_roi, cmap='gray')
    
    # Show processed mask (filled) — use a custom overlay color
    plt.imshow(np.ma.masked_where(processed_mask_roi == 0, processed_mask_roi), 
               cmap=plt.colormaps.get_cmap('spring'), alpha=alpha) # 'magenta', 'spring', 'cool', 'copper', 'viridis', 'plasma', 'Oranges', etc. summer
    
    # Show original mask boundary in red
    plt.contour(boundary_roi, colors='red', linewidths=3.0)

    # plt.title(f"Slice {slice_index} – {operation.capitalize()} Mask Overlay", fontsize=14)
    # plt.title(f"ROI {roi_size}×{roi_size} – {operation.capitalize()} Mask + Boundary", fontsize=14)
    plt.axis('off')
    # Save or show
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    print(f"Overlay ROI saved to {output_path}")
 

def plot_overlay_roi_mask_v2(image_path, mask_path, output_path, slice_index=None,
                          operation='erode', kernel_size=11, iterations=1,
                          overlay_cmap='spring', alpha=0.5, boundary_color='red',
                          roi_size=128):
    # Load NIfTI
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()

    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Normalize MRI
    image_slice = (image_slice - np.min(image_slice)) / np.ptp(image_slice)

    # Apply erosion/dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'erode':
        processed_mask = cv2.erode(mask_slice, kernel, iterations=iterations)
    elif operation == 'dilate':
        processed_mask = cv2.dilate(mask_slice, kernel, iterations=iterations)
    else:
        raise ValueError("Invalid operation. Use 'erode' or 'dilate'.")

    # Crop ROI
    image_roi, processed_mask_roi = extract_roi(image_slice, processed_mask, size=roi_size)
    _, original_mask_roi = extract_roi(image_slice, mask_slice, size=roi_size)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(image_roi, cmap='gray')

    # Overlay processed mask
    plt.imshow(np.ma.masked_where(processed_mask_roi == 0, processed_mask_roi),
               cmap=plt.colormaps.get_cmap(overlay_cmap), alpha=alpha)

    # Extract and plot a single red boundary from the original mask
    contours = measure.find_contours(original_mask_roi, 0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color=boundary_color, linewidth=12, linestyle='dotted')

    # plt.title(f"ROI {roi_size}×{roi_size} – {operation.capitalize()} Mask + Boundary", fontsize=14)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    print(f"Overlay ROI saved to {output_path}")

########################################
## for bbox
def get_all_bounding_boxes_with_ccl_box_rand(mask, bbox_shift=5):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((1, 4))* (-100)#.to(mask_256.device) * (-100), save the mask, 1x4
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((1, 4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                # y_min, x_min = max(0, s[0].start - bbox_shift), max(0, s[1].start - bbox_shift)
                # y_max, x_max = max(0, s[0].stop - 1 + bbox_shift), max(0, s[1].stop - 1 + bbox_shift)
                # y_min = max(0,  s[0].start - random.randint(0, bbox_shift))
                # x_min = max(0,  s[1].start - random.randint(0, bbox_shift))
                # y_max = max(0,  s[0].stop - 1 + random.randint(0, bbox_shift))
                # x_max = max(0,  s[1].stop - 1 + random.randint(0, bbox_shift))

                y_min = max(0, s[0].start + random.choice([ -1]) * random.randint(0, bbox_shift))
                x_min = max(0, s[1].start + random.choice([ -1]) * random.randint(0, bbox_shift))
                y_max = max(0, s[0].stop - 1 + random.choice([ -1]) * random.randint(0, bbox_shift))
                x_max = max(0, s[1].stop - 1 + random.choice([ -1]) * random.randint(0, bbox_shift))



                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## select all bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        sel_bbox = []
        for n in range(len(bboxes[unique_labels[0]])):
            sel_bbox.append(np.array(bboxes[unique_labels[0]][n]))
        sel_bbox = np.array(sel_bbox)
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    
    
    return sel_bbox


def plot_overlay_roi_bbox(image_path, mask_path, output_path, slice_index=None,
                          overlay_cmap='spring', alpha=0.5, boundary_color='red',
                          roi_size=128, bbox_shift=0):
    # Load NIfTI
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()

    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Normalize MRI
    image_slice = (image_slice - np.min(image_slice)) / np.ptp(image_slice)

    # Crop ROI
    image_roi, original_mask_roi = extract_roi(image_slice, mask_slice, size=roi_size)

    ## get bbox
    sel_bbox = get_all_bounding_boxes_with_ccl_box_rand(original_mask_roi, bbox_shift=bbox_shift)
    x_min, x_max = sel_bbox[0], sel_bbox[2]
    y_min, y_max = sel_bbox[1], sel_bbox[3]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(image_roi, cmap='gray')


    # Extract and plot a single red boundary from the original mask
    # Apply erosion to shrink the boundary by 1 pixel
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(original_mask_roi, kernel, iterations=1)

    # Extract contours from the eroded mask
    contours = measure.find_contours(eroded_mask, 0.5)
    # contours = measure.find_contours(original_mask_roi, 0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color=boundary_color, linewidth=4)
    
    # Draw bounding box
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             color='lime', linewidth=4)

    # plt.title(f"ROI {roi_size}×{roi_size} – {operation.capitalize()} Mask + Boundary", fontsize=14)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    print(f"Overlay ROI saved to {output_path}")


## GT bbox overly shift bbox
def plot_overlay_roi_bbox_GT_Shift(image_path, mask_path, output_path, slice_index=None,
                          overlay_cmap='spring', alpha=0.5, boundary_color='red',
                          roi_size=128, bbox_shift=0):
    # Load NIfTI
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()

    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Normalize MRI
    image_slice = (image_slice - np.min(image_slice)) / np.ptp(image_slice)

    # Crop ROI
    image_roi, original_mask_roi = extract_roi(image_slice, mask_slice, size=roi_size)

    ## get GT bbox
    sel_bbox = get_all_bounding_boxes_with_ccl_box_rand(original_mask_roi, bbox_shift=0)
    x_min, x_max = sel_bbox[0], sel_bbox[2]
    y_min, y_max = sel_bbox[1], sel_bbox[3]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(image_roi, cmap='gray')
    # Draw bounding box
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             color='red', linewidth=12, linestyle='dotted')
    

    ## get shifted bbox
    sel_bbox = get_all_bounding_boxes_with_ccl_box_rand(original_mask_roi, bbox_shift=bbox_shift)
    x_min, x_max = sel_bbox[0], sel_bbox[2]
    y_min, y_max = sel_bbox[1], sel_bbox[3]

    # # Plotting
    # plt.figure(figsize=(8, 8))
    # plt.imshow(image_roi, cmap='gray')

    
    # Draw bounding box
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             color='lime', linewidth=4)

    # plt.title(f"ROI {roi_size}×{roi_size} – {operation.capitalize()} Mask + Boundary", fontsize=14)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    print(f"Overlay ROI saved to {output_path}")




def plot_overlay_roi_bbox_crop(image_path, mask_path, output_path, slice_index=None,
                          overlay_cmap='spring', alpha=0.5, boundary_color='red',
                          roi_size=128, bbox_shift=0):
    # Load NIfTI
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()

    if slice_index is None:
        slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index].astype(np.uint8)

    # Normalize MRI
    image_slice = (image_slice - np.min(image_slice)) / np.ptp(image_slice)

    # Crop ROI
    image_roi, original_mask_roi = extract_roi(image_slice, mask_slice, size=roi_size)

    ## get bbox
    sel_bbox = get_all_bounding_boxes_with_ccl_box_rand(original_mask_roi, bbox_shift=bbox_shift)
    x_min, x_max = sel_bbox[0], sel_bbox[2]
    y_min, y_max = sel_bbox[1], sel_bbox[3]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(image_roi, cmap='gray')


    # # Extract and plot a single red boundary from the original mask
    # # Apply erosion to shrink the boundary by 1 pixel
    # kernel = np.ones((3, 3), np.uint8)
    # eroded_mask = cv2.erode(original_mask_roi, kernel, iterations=1)

    # # Extract contours from the eroded mask
    # contours = measure.find_contours(eroded_mask, 0.5)
    # # contours = measure.find_contours(original_mask_roi, 0.5)
    # for contour in contours:
    #     plt.plot(contour[:, 1], contour[:, 0], color=boundary_color, linewidth=4)
    
    # Draw bounding box
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             color=boundary_color, linewidth=12, linestyle='dotted')

    # plt.title(f"ROI {roi_size}×{roi_size} – {operation.capitalize()} Mask + Boundary", fontsize=14)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    print(f"Overlay ROI saved to {output_path}")


# def plot_bbox_overlay(image_path, mask_path, output_path, slice_index=None,
#                       boundary_color='red', box_color='lime', linewidth=2):
#     # Load NIfTI image and mask
#     image_nii = nib.load(image_path)
#     mask_nii = nib.load(mask_path)
#     image = image_nii.get_fdata()
#     mask = mask_nii.get_fdata()

#     # Auto-select slice with most mask content
#     if slice_index is None:
#         slice_index = np.argmax(np.sum(mask, axis=(0, 1)))

#     image_slice = image[:, :, slice_index]
#     mask_slice = mask[:, :, slice_index].astype(np.uint8)

#     # Normalize MRI slice for display
#     image_slice = (image_slice - np.min(image_slice)) / np.ptp(image_slice)

#     # Get bounding box from the mask
#     ys, xs = np.where(mask_slice > 0)
#     if len(xs) == 0 or len(ys) == 0:
#         print("No object found in mask.")
#         return

#     x_min, x_max = xs.min(), xs.max()
#     y_min, y_max = ys.min(), ys.max()

#     # Plot image
#     plt.figure(figsize=(8, 8))
#     plt.imshow(image_slice, cmap='gray')

#     # Plot mask boundary
#     contours = measure.find_contours(mask_slice, 0.5)
#     for contour in contours:
#         plt.plot(contour[:, 1], contour[:, 0], color=boundary_color, linewidth=1.5)

#     # Draw bounding box
#     plt.plot([x_min, x_max, x_max, x_min, x_min],
#              [y_min, y_min, y_max, y_max, y_min],
#              color=box_color, linewidth=linewidth)

#     plt.title(f"Bounding Box and Ground Truth – Slice {slice_index}")
#     plt.axis('off')
#     # plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
#     plt.show()
#     print(f"Bounding box overlay saved to {output_path}")


########################################

data_path = "/home/gxu/proj1/lesionSeg/dataset/utswdataset_gp"
## for bbox
# img_path = os.path.join(data_path, "93228290/93228290_20240723_mr.nii.gz")
# gt_path = os.path.join(data_path, "93228290/93228290_20240723_gt_itv_combined.nii.gz")
## for GT mask
img_path = os.path.join(data_path, "93228290/93228290_20240821_mr.nii.gz")
gt_path = os.path.join(data_path, "93228290/93228290_20240821_gt_itv_combined.nii.gz")
save_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/aug_prompt_v2"



# # plot erode mask
# kernel_sizes = [3, 5, 7, 9, 11, 17]
# for n in kernel_sizes:
#     save_name = os.path.join(save_path, "roi_erode_"+str(n)+".png")
#     plot_overlay_roi_mask_v2(img_path, gt_path, save_name, kernel_size=n, slice_index=None, operation="erode", boundary_color='yellow')
#     # break

kernel_sizes = [3, 5, 7, 9, 11, 17]
for n in kernel_sizes:
    save_name = os.path.join(save_path, "roi_dilate_"+str(n)+".png")
    plot_overlay_roi_mask_v2(img_path, gt_path, save_name, kernel_size=n, slice_index=None, operation="dilate", boundary_color='yellow')
    # break


# # get bbox and shift bbox
# bbox_shifts = [2, 4, 6, 8, 10, 12]
# for bbox_shift in bbox_shifts:
#     save_name = os.path.join(save_path, "roi_bbox_aug_"+str(bbox_shift)+".png")
#     plot_overlay_roi_bbox_GT_Shift(img_path, gt_path, save_name, slice_index=None,
#                             overlay_cmap='spring', alpha=0.5, boundary_color='yellow',
#                             roi_size=128, bbox_shift=bbox_shift) # bbox_shift
#     # break

bbox_shifts = [2, 4, 6, 8, 10, 12]
for bbox_shift in bbox_shifts:
    save_name = os.path.join(save_path, "roi_bbox_aug_"+str(bbox_shift)+"_small.png")
    plot_overlay_roi_bbox_GT_Shift(img_path, gt_path, save_name, slice_index=None,
                            overlay_cmap='spring', alpha=0.5, boundary_color='yellow',
                            roi_size=128, bbox_shift=bbox_shift) # bbox_shift

# ## plot original mask
# save_name = os.path.join(save_path, "original_mask.png")
# plot_overlay_roi_mask_v2(img_path, gt_path, save_name, kernel_size=1, slice_index=None, operation="erode", boundary_color='yellow')

# # plot original bbox crop  
# save_name = os.path.join(save_path, "original_bbox.png")
# plot_overlay_roi_bbox_crop(img_path, gt_path, save_name, slice_index=None,
#                         overlay_cmap='spring', alpha=0.5, boundary_color='red',
#                         roi_size=128, bbox_shift=0)

