import os
import numpy as np
import matplotlib.pyplot as plt

def extract_mean_std_from_line(line):
    # Extract mean and std from the given line (format: value±value)
    try:
        mean, std = line.split('±')
        mean = float(mean)
        std = float(std)
        return mean, std
    except ValueError:
        return None, None


def read_metric_values(directory, idx):
    """
    idx=0: dice
    idx=1: NSD
    idx=2: HD95
    idx=3: ASD
    """
    for file_name in sorted(os.listdir(directory)):
        if "format_test_mean" in file_name:
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 3: # read the 3rd line
                    mean_dice, std_dice = extract_mean_std_from_line(lines[2].split('\t')[idx])
                    if mean_dice is not None:
                        return mean_dice, std_dice


## for the method 
root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir/paper_used_model/Sam2_Tiny_Promp_Aug_3Ch"
# methods = ['AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5', 'AugBox_Mask_FT_Img_Msk_Prmp_3ch_Rand5', 'check_AugBox_NoMask_FT_Img_Msk_Prmp_3ch_Rand5', 
#            'Box_Mask_FT_Img_Msk_Prmp_3ch_Rand0', 'Box_NoMask_FT_Img_Msk_Prmp_3ch_Rand0', ] # 'AugMask_FT_Img_Msk_Prmp_3ch_Rand5'
methods = ['AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5',  'check_AugBox_NoMask_FT_Img_Msk_Prmp_3ch_Rand5', 
            'Box_Mask_FT_Img_Msk_Prmp_3ch_Rand0', 'Box_NoMask_FT_Img_Msk_Prmp_3ch_Rand0', ] # 'AugMask_FT_Img_Msk_Prmp_3ch_Rand5'
method_dirs = dict()

IDX = 3
TYPE_METRIC = ["Dice", "NSD", "HD95", "ASD"]

## collect all path
for cnt, method in enumerate(methods, start=1):
    for num in range(0, 11):
        # method_dir = os.path.join(root_path, method, "dataset_hr/itv_eval_result_bbox_shift"+str(num)) # 
        method_dir = os.path.join(root_path, method, "eval_result_bbox_shift"+str(num)) # 
        if "method"+str(cnt) in method_dirs:
            method_dirs["method"+str(cnt)].append(method_dir)
        else:
            method_dirs["method"+str(cnt)] = [method_dir]

## read dice (mean and std)
method_keys = method_dirs.keys()
dice_all_values_mean = np.zeros((len(methods), 11))
dice_all_values_std = np.zeros((len(methods), 11))
dice_one_method_mean = []
dice_one_method_std = []
for num_row, key in enumerate(method_dirs):
    for num_col, path_res in enumerate(method_dirs[key]):
        # print(path_res)
        ## read dice
        dice_all_values_mean[num_row, num_col], dice_all_values_std[num_row, num_col] = read_metric_values(path_res, idx=IDX)
## remove the shift=0
dice_all_values_mean = dice_all_values_mean[:, 1:]
dice_all_values_std = dice_all_values_std[:, 1:]

print(dice_all_values_mean.shape)
print(dice_all_values_std.shape)



# Plot the results
plt.figure(figsize=(8, 6), dpi=300)
# plt.figure(figsize=(10, 6))
# methods_legend = ['Aug-BBox-Mask', 'NoAug-BBox-Mask', 'Aug-BBox-NoMask', 'NoAug-BBox-NoMask']
# methods_legend = ['AugBBox+AugMask', 'AugBBox+Mask', 'AugBBox+NoMask', 'BBox+Mask', 'BBox+NoMask', ] # 'NoBBox+AugMask'
# methods_legend = ['AugBBox+AugMask', 'AugBBox+NoMask', 'BBox+Mask', 'BBox+NoMask', ] # 'NoBBox+AugMask'
methods_legend = ['AugBBox+AugMask',  'AugBBox', 'BBox+Mask',  'BBox', ] # 'NoBBox+AugMask'
for i, method in enumerate(methods_legend):
    x = np.arange(1, 11)  # 10 points for each method
    plt.errorbar(x, dice_all_values_mean[i], yerr=dice_all_values_std[i], fmt='o-', label=method, capsize=5)

plt.title('Comparison of Mean '+TYPE_METRIC[IDX]+' with Standard Deviation', fontsize=12) # , fontweight='bold'
plt.xlabel('BBox Random Expansion Range', fontsize=12)
plt.ylabel('Mean '+TYPE_METRIC[IDX], fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Minimize white space around the plot
plt.tight_layout()
# Save with high resolution and crop blank regions
# plt.savefig('/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/comparison_dice_methods.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig('/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/compare_diff_aug_fourv2/comparison_'+TYPE_METRIC[IDX]+'_methods_aug_gp_itv.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()