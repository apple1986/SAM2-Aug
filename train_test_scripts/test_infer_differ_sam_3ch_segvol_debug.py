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
# from radioa.utils.io import binarize_gt

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
###########################
## set parameters
parser = argparse.ArgumentParser(
    description="Run inference on validation set with MedSAM2"
)
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

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

#######################################
"""different sam-based models"""
from pathlib import Path
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.nn.functional as F
from typing import TypeVar
import torchio as tio
from itertools import product
import nibabel as nib


from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep
from radioa.utils.SAMMed3D_segment_anything.build_sam3D import build_sam3D_vit_b_ori
from radioa.utils.SAMMed3D_segment_anything.modeling.sam3D import Sam3D
from radioa.utils.image import get_crop_pad_params_from_gt_or_prompt
from radioa.utils.resample import get_current_spacing_from_affine, resample_to_shape, resample_to_spacing
from radioa.utils.transforms import resample_to_spacing_sparse
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
# from radioa.model import segvol

import radioa.prompts.prompter as prmp

######################################
from argparse import Namespace
from pathlib import Path
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from pathlib import Path
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from radioa.model.inferer import Inferer
from radioa.prompts.prompt import Boxes3D, Points, PromptStep
from radioa.utils.SegVol_segment_anything.network.model import SegVol
import torch
import os
import sys
from radioa.utils.SegVol_segment_anything.monai_inferers_utils import (
    build_binary_points,
    build_binary_cube,
    logits2roi_coor,
    sliding_window_inference,
)
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from copy import deepcopy

from radioa.utils.SegVol_segment_anything import sam_model_registry
from radioa.utils.paths import get_model_path
from radioa.utils.transforms import resample_to_shape_sparse


class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d


class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d


class ForegroundNormalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d

    def normalize(self, ct_narray):
        ct_voxel_ndarray = deepcopy(ct_narray)
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ### transform ###
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray


def load_segvol(checkpoint_path):
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Add this directory to the sys.path to allow relative path to clip checkpoint
    if script_directory not in sys.path:
        sys.path.append(script_directory)

    model_ckpt = get_model_path() / "SegVol_v1.pth"
    args = Namespace(
        test_mode=True,
        resume=checkpoint_path,
        infer_overlap=0.5,
        spatial_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        clip_ckpt=str(model_ckpt),  # This might not work if not running the .py in the base dir
    )

    gpu = 0

    sam_model = sam_model_registry["vit"](args=args)

    segvol_model = SegVol(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_ckpt=args.clip_ckpt,
        roi_size=args.spatial_size,
        patch_size=args.patch_size,
        test_mode=args.test_mode,
    ).cuda()
    segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[gpu])

    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        loc = "cuda:{}".format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        segvol_model.load_state_dict(checkpoint["model"], strict=False)
    segvol_model.eval()

    return segvol_model


class SegVolInferer(Inferer):

    pass_prev_prompts = True
    dim = 3
    supported_prompts = ("box", "point")

    def __init__(self, checkpoint_path, device="cuda"):
        if device != "cuda":
            raise RuntimeError("segvol can only be run on cuda.")
        super().__init__(checkpoint_path, device)
        self.prev_mask = None
        self.inputs = None
        self.mask_threshold = 0
        self.infer_overlap = 0.5
        self.start_coord = None
        self.end_coord = None

        self.spatial_size = (32, 256, 256)

        self.transform_img = transforms.Compose(
            [
                transforms.Orientationd(
                    keys=["image"], axcodes="RAS"
                ),  # Doesn't actually do anything since metadata is discarded. Kept for comparability to original
                ForegroundNormalization(keys=["image"]),
                DimTranspose(keys=["image"]),
                MinMaxNormalization(),
                transforms.SpatialPadd(keys=["image"], spatial_size=(32, 256, 256), mode="constant"),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.transform_seg = transforms.Compose(
            [
                transforms.Orientationd(
                    keys=["image", "seg"], axcodes="RAS"
                ),  # Doesn't actually do anything since metadata is discarded. Kept for comparability to original
                DimTranspose(keys=["image", "seg"]),
                transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(32, 256, 256), mode="constant"),
                transforms.CropForegroundd(keys=["image", "seg"], source_key="image"),
                transforms.ToTensord(keys=["image", "seg"]),
            ]
        )
        self.zoom_out_transform = transforms.Resized(
            keys=["image"], spatial_size=self.spatial_size, mode="nearest-exact"
        )
        self.img_loader = transforms.LoadImage()

    def load_model(self, checkpoint_path, device):
        return load_segvol(checkpoint_path)

    # def set_image(self, img_path):
    #     if self._image_already_loaded(img_path=img_path):
    #         return
    #     img_nib = load_any_to_nib(img_path)
    #     self.orig_affine = img_nib.affine
    #     self.orig_shape = img_nib.shape

    #     self.img, self.img_zoom_out, self.start_coord, self.end_coord = self.transform_to_model_coords_dense(
    #         img_nib, is_seg=False
    #     )
    #     self.cropped_shape = self.img.shape
    #     self.loaded_image = img_path

    def set_image(self, img_np) -> None:
        
        img_np = img_np #* 255
        self.orig_affine = np.array([[-0.88999999,  0.,          0.,        -0.,        ],
                            [ 0.,         -0.88999999,  0.,         -0.,        ],
                            [ 0. ,         0. ,         3.  ,        0.   ,     ],
                            [ 0. ,         0. ,         0.  ,        1.   ,     ]])
        
        self.orig_shape = (512, 512, 128)

        self.img, self.inv_trans_dense = self.transform_to_model_coords_dense(img_np, is_seg=False)

        self.img, self.img_zoom_out, self.start_coord, self.end_coord = self.transform_to_model_coords_dense(
        img_nib, is_seg=False)

        # # clip image to 0.5% - 99.5%
        # self.global_min = np.percentile(self.img, 0.5)
        # self.global_max = np.percentile(self.img, 99.5)

        # # Clip the image array
        # self.img = np.clip(self.img, self.global_min, self.global_max)
        self.img = self.img #* 255.0

    def transform_to_model_coords_dense(
        self, nifti: str | Path | nib.Nifti1Image, is_seg: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(nifti, (str, Path)):
            nifti = load_any_to_nib(nifti)

        if is_seg:
            # raise RuntimeError('An image must first be set before segmentations can be transformed')
            return np.zeros((10, 10, 10))  # Disgusting fix since groundtruth_model isn't needed

        item = {}
        # generate ct_voxel_ndarray
        img = nifti.get_fdata()
        img = np.expand_dims(
            img, axis=0
        )  # Ensure image is in CDWH (spatial dimensions will be assumed to be RAS anyway)

        # transform
        if not is_seg:
            item["image"] = img
            item = self.transform_img(item)
        else:
            dummy_img = np.zeros_like(self.img)
            dummy_img[
                self.start_coord[0] : self.end_coord[0],
                self.start_coord[1] : self.end_coord[1],
                self.start_coord[2] : self.end_coord[2],
            ] = 1  # used for crop transform; must crop the same way as the image was cropped.
            item["image"] = dummy_img
            item["seg"] = img

            item = self.transform_seg(item)
            item["image"] = item["seg"]
            del item["seg"]
            # now make appropriate for zoom out transform

        start_coord = item["foreground_start_coord"]  # Store metadata for later inveresion of transformations
        end_coord = item["foreground_end_coord"]

        item_zoom_out = self.zoom_out_transform(item)
        item["zoom_out_image"] = item_zoom_out["image"]
        image, image_zoom_out = item["image"].float().unsqueeze(0), item["zoom_out_image"].float().unsqueeze(0)
        image_single = image[0, 0]

        img, img_zoom_out = image_single, image_zoom_out
        return img, img_zoom_out, start_coord, end_coord

    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        # OrientationD: Do nothing

        # DimTranspose
        coords = coords[[2, 1, 0]]  # Swap the first and last coordinates

        # SpatialPad
        # Calculate padding amounts
        shape_after_dimtranspose = self.orig_shape[::-1]

        # Calculate padding needed for each axis
        total_pads = np.maximum(np.array(self.spatial_size) - np.array(shape_after_dimtranspose), 0)
        pad_starts = total_pads // 2

        # Adjust point coordinates for padding
        coords = coords + pad_starts

        # CropForegroundd
        coords = coords - self.start_coord

        # Ensure that coords lie within model patch - can occur if 'background' is segmented as gt
        coords = np.minimum(coords, self.cropped_shape)
        coords = np.maximum(coords, 0)

        # 5. Resized (if applicable)
        zoomed_out_coords = resample_to_shape_sparse(coords, self.cropped_shape, self.spatial_size, round=True)

        return zoomed_out_coords

    def preprocess_prompt(self, prompt, prompt_type, text_prompt=None):
        """
        Preprocessing steps:
            - Modify in line with the volume cropping
            - Modify in line with the interpolation
            - Collect into a dictionary of slice:slice prompt
        """
        # text_single, box_single, binary_cube_resize, points_single, binary_points_resize = prompt

        # Clean up
        box_single = points_single = binary_cube = binary_points = None

        if prompt_type == "point":
            coords, labels = prompt.coords, prompt.labels
            # # coords = coords[:,::-1]
            # # Transform in line with cropping # These transforms were moved to transform_to_model_coordinates_sparse
            # coords = coords - self.start_coord
            # coords = np.maximum(coords, 0)
            # coords = np.minimum(coords, self.end_coord)
            # coords = np.round(
            #     coords * np.array(self.spatial_size) / np.array(self.cropped_shape)
            # )  # Transform in line with resizing

            coords, labels = torch.from_numpy(coords).float(), torch.from_numpy(labels).float()
            points_single = (coords.unsqueeze(0).cuda(), labels.unsqueeze(0).cuda())
            binary_points_resize = build_binary_points(coords, labels, self.spatial_size)
            binary_points = F.interpolate(
                binary_points_resize.unsqueeze(0).unsqueeze(0).float(), size=self.cropped_shape, mode="nearest"
            )[0][0]
        if prompt_type == "box":
            # # Transform box in line with image transformations # These transforms were moved to transform_to_model_coordinates_sparse
            # box_single = np.array(prompt.bbox)  # Change from pair of points to 2x3 array
            # box_single = box_single - self.start_coord  # transform in line with cropping
            # box_single[0] = np.maximum(box_single[0], 0)
            # box_single[1] = np.minimum(box_single[1], self.end_coord)
            # box_single = np.round(
            #     box_single * np.array(self.spatial_size) / np.array(self.cropped_shape)
            # )  # Transform in line with resizing

            box_single = np.array(prompt.bbox)  # Change from pair of points to 2x3 array
            box_single = box_single.flatten()
            box_single = torch.from_numpy(box_single).unsqueeze(0).float().cuda()

            # Obtain binary cube
            binary_cube_resize = build_binary_cube(box_single, self.spatial_size)
            binary_cube = F.interpolate(
                binary_cube_resize.unsqueeze(0).unsqueeze(0).float(), size=self.cropped_shape, mode="nearest"
            )[0][0]

        text_single = text_prompt
        prompt = text_single, box_single, binary_cube, points_single, binary_points
        return prompt

    def preprocess_img(self, img: np.ndarray):  # image preprocessing is handled in set_image
        pass

    @torch.no_grad()
    def segment(self, image_single, image_single_resize, prompt, prompt_type):
        # zoom-out inference:
        text_single, box_single, binary_cube, points_single, binary_points = prompt
        logits_global_zoom_out = self.model(
            image_single_resize.cuda(), text=text_single, boxes=box_single, points=points_single
        )

        # resize back global logits
        logits_global_zoom_out = F.interpolate(logits_global_zoom_out.cpu(), size=self.cropped_shape, mode="nearest")[
            0
        ][0]

        # zoom-in inference:
        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(self.spatial_size, logits_global_zoom_out)
        if min_d is None:
            res = (None, None, None, None, logits_global_zoom_out)
            return res
            # raise RuntimeError("Fail to detect foreground!")

        # Crop roi
        image_single_cropped = (
            image_single[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1].unsqueeze(0).unsqueeze(0)
        )
        global_preds = (
            torch.sigmoid(logits_global_zoom_out[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1]) > 0.5
        ).long()

        prompt_reflection = None
        if prompt_type == "box":
            binary_cube_cropped = binary_cube[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1]
            prompt_reflection = (
                binary_cube_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0),
            )
        if prompt_type == "point":
            binary_points_cropped = binary_points[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1]
            prompt_reflection = (
                binary_points_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0),
            )

        ## inference
        logits_single_cropped = sliding_window_inference(
            image_single_cropped.cuda(),
            prompt_reflection,
            self.spatial_size,
            1,
            self.model,
            self.infer_overlap,
            text=text_single,
            use_box=(prompt_type == "box"),
            use_point=(prompt_type == "point"),
        )
        logits_single_cropped = logits_single_cropped.cpu().squeeze()
        logits_global_zoom_in = logits_global_zoom_out.clone()
        logits_global_zoom_in[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1] = logits_single_cropped

        res = (image_single, points_single, box_single, logits_global_zoom_out, logits_global_zoom_in)

        return res

    def postprocess_mask(self, mask, return_logits):
        """
        Postprocessing steps:
            - TODO
        """
        if not return_logits:
            mask = (mask > 0).cpu().numpy().astype(np.uint8)

        return mask

    def inv_trans_dense(self, mask: np.ndarray) -> nib.Nifti1Image:
        # Invert transform
        shape_after_dimtranspose = self.orig_shape[::-1]
        # undo crop foreground
        padded_segmenation_shape = np.maximum(self.spatial_size, shape_after_dimtranspose)
        padded_segmentation = np.zeros(
            padded_segmenation_shape, dtype=np.uint8
        )  # Should be created in 32,256,256 shape
        padded_segmentation[
            self.start_coord[0] : self.end_coord[0],
            self.start_coord[1] : self.end_coord[1],
            self.start_coord[2] : self.end_coord[2],
        ] = mask  #  stick into 32,256,256 shape then undo pad # ToDo needs fixing, try MS instances:

        # undo spatial pad
        total_pads = np.maximum(np.array(self.spatial_size) - np.array(shape_after_dimtranspose), 0)
        pad_starts = total_pads // 2

        transposed_segmentation = padded_segmentation[
            pad_starts[0] : shape_after_dimtranspose[0] + pad_starts[0],
            pad_starts[1] : shape_after_dimtranspose[1] + pad_starts[1],
            pad_starts[2] : shape_after_dimtranspose[2] + pad_starts[2],
        ]

        # undo dim_transpose
        segmentation = np.transpose(transposed_segmentation, (2, 1, 0))

        segmentation = nib.Nifti1Image(segmentation, self.orig_affine)

        return segmentation

    def predict(
        self,
        prompt: PromptStep | Boxes3D,
        text_prompt=None,
        return_logits=False,
        prev_seg=None,
        promptstep_in_model_coord_system=False,
        seed=1,
    ):
        if self.loaded_image is None:
            raise RuntimeError("Must first set image!")

        if not isinstance(prompt, (Boxes3D, PromptStep)):
            raise TypeError(
                "Prompts must be 3d bounding boxes or points, and must be supplied as an instance of Boxes3D or PromptStep"
            )

        prompt_type = "box" if isinstance(prompt, Boxes3D) else "point"

        if prompt_type == "point":
            torch.manual_seed(
                seed
            )  # New points are sampled in the zoom-in section of zoom-out zoom-in inference leaving some randomness even after a point prompt is fixed.

        image_single, image_single_resize = self.img, self.img_zoom_out

        prompt = deepcopy(prompt)
        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)
        prompt = self.preprocess_prompt(prompt, prompt_type, text_prompt)

        res = self.segment(image_single, image_single_resize, prompt, prompt_type)

        segmentation = self.postprocess_mask(res[-1], return_logits)

        segmentation_model_arr = segmentation

        # Turn into Nifti object in original space
        segmentation_orig_nib = self.inv_trans_dense(segmentation)

        low_res_logits = None  # low_res_logits aren't easily accessed nor used since segvol isn't interactive

        return segmentation_orig_nib, low_res_logits, segmentation_model_arr

######################################
def binarize_gt_ap(gt_mask, affine_matrix, label_of_interest=1) -> nib.Nifti1Image:
    """
    Creates a binary mask from a multi-class groundtruth in the same spacing.
    """

    binary_gt = np.where(gt_mask == label_of_interest, 1, 0)
    binary_gt = nib.Nifti1Image(binary_gt.astype(np.uint8), affine_matrix)

    return binary_gt

#######################################
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
    model_infer = SegVolInferer(checkpoint_path=checkpoint_path, device=device)

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
