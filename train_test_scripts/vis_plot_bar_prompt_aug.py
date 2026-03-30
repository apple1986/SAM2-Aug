import os
import numpy as np
import matplotlib.pyplot as plt

## 
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
        if "format_test_mean_dice_hd_nsd_asd5" in file_name:
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 3: # read the 3rd line
                    mean_dice, std_dice = extract_mean_std_from_line(lines[2].split('\t')[idx])
                    if mean_dice is not None:
                        return mean_dice, std_dice

def plot_bar_input(data, metric_type="Dice"):
    tumor_types = ['GTV', 'ITV', 'PTV', 'ALL']
    x = np.arange(len(tumor_types))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plotting two sets of bars for each row
    ax.bar(x - width/2, data[0], width, label='Input 1', color='skyblue')
    ax.bar(x + width/2, data[1], width, label='Input 2', color='lightcoral')

    ax.set_ylabel(metric_type)
    ax.set_xlabel('Tumor Type')
    ax.set_title(f'{metric_type} for Different Tumor Types')
    ax.set_xticks(x)
    ax.set_xticklabels(tumor_types)
    ax.legend(['curMR', 'curMR+preMR+preGT'])

    plt.show()

def plot_rectplot_with_std(mean_data, std_data, metric_type="Dice"):
    tumor_types = ['GTV', 'ITV', 'PTV', 'ALL']
    x = np.arange(len(tumor_types))  # the label locations
    width = 0.18  # narrower bar width to fit four bars per group

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plotting four groups of bars
    rects1 = ax.bar(x - 1.5 * width, mean_data[0], width, yerr=std_data[0], 
                    label='BBox', color='skyblue', capsize=4)
    rects2 = ax.bar(x - 0.5 * width, mean_data[1], width, yerr=std_data[1], 
                    label='BBox+Mask', color='lightgreen', capsize=4)
    rects3 = ax.bar(x + 0.5 * width, mean_data[2], width, yerr=std_data[2], 
                    label='AugBBox', color='salmon', capsize=4)
    rects4 = ax.bar(x + 1.5 * width, mean_data[3], width, yerr=std_data[3], 
                    label='AugBBox+AugMask', color='plum', capsize=4)

    # Y-axis limit and labels
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel(metric_type, fontsize=12)
    ax.set_xlabel('Tumor Type', fontsize=12)
    ax.set_title(f'{metric_type} for Different Tumor Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tumor_types, fontsize=11)

    # Legend
    ax.legend(loc='lower right', fontsize=10)

    # Grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    return fig


## set path
## for the method 
root_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2/work_dir/paper_used_model/Sam2_Tiny_Promp_Aug_3Ch"
methods = ['Box_NoMask_FT_Img_Msk_Prmp_3ch_Rand0', 'Box_Mask_FT_Img_Msk_Prmp_3ch_Rand0',
           'check_AugBox_NoMask_FT_Img_Msk_Prmp_3ch_Rand5', 'AugBox_AugMask_FT_Img_Msk_Prmp_3ch_Rand5']
tv_types = ['gtv', 'itv', 'ptv', 'alltv'] # 'AugMask_FT_Img_Msk_Prmp_3ch_Rand5'
method_dirs = dict()
## collect all path
for cnt, method in enumerate(methods, start=1):
    for num in range(0, len(tv_types)):
        method_dir = os.path.join(root_path, method, "dataset_hr/eval_result_"+tv_types[num])
        if "method"+str(cnt) in method_dirs:
            method_dirs["method"+str(cnt)].append(method_dir)
        else:
            method_dirs["method"+str(cnt)] = [method_dir]

## read dice (mean and std)
method_keys = method_dirs.keys()
dice_all_values_mean = np.zeros((len(methods), len(tv_types)))
dice_all_values_std = np.zeros((len(methods), len(tv_types)))
dice_one_method_mean = []
dice_one_method_std = []
for num_row, key in enumerate(method_dirs):
    for num_col, path_res in enumerate(method_dirs[key]):
        # print(path_res)
        ## read dice
        dice_all_values_mean[num_row, num_col], dice_all_values_std[num_row, num_col] = read_metric_values(path_res, idx=0)

print(dice_all_values_mean.shape)
print(dice_all_values_std.shape)
## plot
fig = plot_rectplot_with_std(dice_all_values_mean, dice_all_values_std)
# Minimize white space around the plot
plt.tight_layout()
# Save with high resolution and crop blank regions
# plt.savefig('/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/comparison_dice_methods.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig('/home/gxu/proj1/lesionSeg/utswlits3d_v2/paper_vis/comparison_diff_prompt_hr.png', dpi=300, bbox_inches='tight', pad_inches=0.1)