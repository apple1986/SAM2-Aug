
import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from model_factory.feature3D_extractor import Feature_DiFPN, Feature_DiFPN_Focus, Feature_DiFPN_Re

## conf and build pre-trained model
def load_pretrain_model(model_cfg, pretrain_model_path, device):
    """
    model_cfg = "sam2_hiera_t.yaml"
    pretrain_model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt"
    """
    sam2_model = build_sam2(model_cfg, pretrain_model_path, device=device, apply_postprocessing=True)
    return sam2_model

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

        # Memory block
        for param in self.sam2_model.memory_encoder.parameters():
            param.requires_grad = True
        for param in self.sam2_model.memory_attention.parameters():
            param.requires_grad = True
            
        # freeze image encoder
        for param in self.sam2_model.image_encoder.parameters():
            param.requires_grad = True

        self.feat_fuse = Conv3d_Sam2_feat_fuse(ks=3, stride=1, pad=1)
        

    def forward(self, image, box, masks=None, feat_cnn=None,):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        masks: Bx4
        """
        _features = self._image_encoder(image)
        ## add 3d information
        if feat_cnn is not None:
            _features["high_res_feats"] = self.feat_fuse(feat_cnn[0], feat_cnn[1], _features["high_res_feats"])
        ## pass the features
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
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
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(), # ([1, 256, 64, 64])
            sparse_prompt_embeddings=sparse_embeddings, # 2x3x256
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits # ([2, 1, 256, 256])
    
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
    
    ## add a new 3D feature fusion moduel


class Conv3d_Sam2_feat_fuse(nn.Module):
    def __init__(self, ks=3, stride=1, pad=1):
        super().__init__()
        self.feat_fuse_1 = nn.Sequential(nn.Conv2d(160, 64, kernel_size=ks, stride=stride, padding=pad),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 32, kernel_size=ks, stride=1, padding=pad),
                                            nn.BatchNorm2d(32),
                                            )
        self.feat_fuse_2 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=ks, stride=stride, padding=pad),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 64, kernel_size=ks, stride=1, padding=pad),
                                            nn.BatchNorm2d(64),
                                            )

    def forward(self, feat_cnn1, feat_cnn2, feat_sam2):
        feat_trans1 = feat_sam2[0] # Bx32x256x256
        feat_trans2 = feat_sam2[1] # Bx64x128x128

        feat_fuse1 = torch.cat((feat_trans1, feat_cnn1), dim=1)
        feat_fuse2 = torch.cat((feat_trans2, feat_cnn2), dim=1)
        ## do feature fusion
        feat_fuse1 = self.feat_fuse_1(feat_fuse1)
        feat_fuse1 = feat_fuse1 + feat_trans1
        feat_fuse2 = self.feat_fuse_2(feat_fuse2)
        feat_fuse2 = feat_fuse2 + feat_trans2

        feat_sam2[0] = feat_fuse1
        feat_sam2[1] = feat_fuse2

        return feat_sam2
####################################
## version 2: reduce the slice number, only care the slices that have bbox
## only pass the second features
class Conv3d_Sam2_feat_fuse_reduce(nn.Module):
    def __init__(self, ks=3, stride=1, pad=1):
        super().__init__()
        self.feat_fuse_1 = nn.Sequential(nn.Conv2d(48, 64, kernel_size=ks, stride=stride, padding=pad),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 32, kernel_size=ks, stride=1, padding=pad),
                                            nn.BatchNorm2d(32),
                                            )
        self.feat_fuse_2 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=ks, stride=stride, padding=pad),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 64, kernel_size=ks, stride=1, padding=pad),
                                            nn.BatchNorm2d(64),
                                            )

    def forward(self, feat_cnn1, feat_cnn2, feat_sam2):
        feat_trans1 = feat_sam2[0] # Bx32x256x256
        feat_trans2 = feat_sam2[1] # Bx64x128x128

        feat_fuse1 = torch.cat((feat_trans1, feat_cnn1), dim=1)
        feat_fuse2 = torch.cat((feat_trans2, feat_cnn2), dim=1)
        ## do feature fusion
        feat_fuse1 = self.feat_fuse_1(feat_fuse1)
        feat_fuse1 = feat_fuse1 + feat_trans1
        feat_fuse2 = self.feat_fuse_2(feat_fuse2)
        feat_fuse2 = feat_fuse2 + feat_trans2

        feat_sam2[0] = feat_fuse1
        feat_sam2[1] = feat_fuse2

        return feat_sam2

class MedSAM2_Re(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # unfreeze prompt encoder
        # for param in self.sam2_model.sam_prompt_encoder.parameters():
        #     param.requires_grad = True
        # unfreeze mask decoder
        for param in self.sam2_model.sam_mask_decoder.parameters():
            param.requires_grad = True

        # # Memory block
        # for param in self.sam2_model.memory_encoder.parameters():
        #     param.requires_grad = True
        # for param in self.sam2_model.memory_attention.parameters():
        #     param.requires_grad = True
            
        # unfreeze image encoder
        for param in self.sam2_model.image_encoder.parameters():
            param.requires_grad = True

        self.feat_fuse = Conv3d_Sam2_feat_fuse_reduce(ks=3, stride=1, pad=1)
        

    def forward(self, image, box, masks=None, feat_cnn=None,):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        masks: Bx4
        """
        _features = self._image_encoder(image)
        ## add 3d information
        if feat_cnn is not None:
            _features["high_res_feats"] = self.feat_fuse(feat_cnn[0], feat_cnn[1], _features["high_res_feats"])
        ## pass the features
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
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
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(), # ([1, 256, 64, 64])
            sparse_prompt_embeddings=sparse_embeddings, # 2x3x256
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits # ([2, 1, 256, 256])
    
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

class MedSAM2_Focus(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # unfreeze prompt encoder
        # for param in self.sam2_model.sam_prompt_encoder.parameters():
        #     param.requires_grad = True
        # unfreeze mask decoder
        for param in self.sam2_model.sam_mask_decoder.parameters():
            param.requires_grad = True

        # # Memory block
        # for param in self.sam2_model.memory_encoder.parameters():
        #     param.requires_grad = True
        # for param in self.sam2_model.memory_attention.parameters():
        #     param.requires_grad = True
            
        # unfreeze image encoder
        for param in self.sam2_model.image_encoder.parameters():
            param.requires_grad = True

        self.feat_fuse = Conv3d_Sam2_feat_fuse_reduce(ks=3, stride=1, pad=1)
        

    def forward(self, image, box, masks=None, feat_cnn=None,):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        masks: Bx4
        """
        _features = self._image_encoder(image)
        ## add 3d information
        if feat_cnn is not None:
            _features["high_res_feats"] = self.feat_fuse(feat_cnn[0], feat_cnn[1], _features["high_res_feats"])
        ## pass the features
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
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
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(), # ([1, 256, 64, 64])
            sparse_prompt_embeddings=sparse_embeddings, # 2x3x256
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits # ([2, 1, 256, 256])
    
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

## define a sam2 with 3d adaptor
def load_medsam2_3d_apt(model_cfg, pretrain_model_path, device):
    # model_cfg = "sam2_hiera_t.yaml"
    # pretrain_model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt"
    # device = "cuda:1"
    sam2_model = build_sam2(model_cfg, pretrain_model_path, device=device, apply_postprocessing=True)
    feat_extractor_3d = Feature_DiFPN(in_channels=3).to(device) # input 512x512x128
    medsam_model = MedSAM2(model=sam2_model).to(device) 

    return feat_extractor_3d, medsam_model

def load_medsam2_3d_apt_re(model_cfg, pretrain_model_path, device, use_center_crop=False):
    # model_cfg = "sam2_hiera_t.yaml"
    # pretrain_model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt"
    # device = "cuda:1"
    sam2_model = build_sam2(model_cfg, pretrain_model_path, device=device, apply_postprocessing=True)
    if use_center_crop is True:
        feat_extractor_3d = Feature_DiFPN_Focus(in_channels=3).to(device) # input 256x256x8
    else:
        feat_extractor_3d = Feature_DiFPN_Re(in_channels=3).to(device) # input 512x512x8
    medsam_model = MedSAM2_Re(model=sam2_model).to(device) 

    return feat_extractor_3d, medsam_model


if __name__ == "__main__":
    from model_factory.feature3D_extractor import Feature_DiFPN
    model_cfg = "sam2_hiera_t.yaml"
    pretrain_model_path = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt"
    device = "cuda:1"
    sam2_model = build_sam2(model_cfg, pretrain_model_path, device=device, apply_postprocessing=True)
    feat_extractor_3d = Feature_DiFPN_Focus(in_channels=3).to(device)
    medsam_model = MedSAM2_Re(model=sam2_model).to(device)
    img = torch.randn((2, 3, 1024, 1024)).to(device)
    img_3d = torch.randn((2, 3, 256, 256, 8)).to(device)
    bbox = torch.tensor([[100, 100, 150, 150], [-200, -200, -200, 200]]).to(device)
    feat_cnn = feat_extractor_3d(img_3d)

    pd_logits = medsam_model(img, bbox, feat_cnn=feat_cnn)
    print(pd_logits.shape)



