import torch
import torch.nn as nn
from torchvision.ops import feature_pyramid_network, FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models import resnet50
from torchvision.models.detection import generalized_rcnn, fasterrcnn_resnet50_fpn, rpn, anchor_utils
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from omegaconf import OmegaConf
import sys
sys.path.append('/Midgard/home/tibbe/3damodal/3DAmodal/datasets')
from dataloader import get_dataloader
from utils import get_obj_from_str
from aisformer_orig import AISFormer as AISFormer_orig

from detectron2.modeling import FPN




def binary_CE(pred, target):
    """
        Compute the loss between a prediction mask and a ground truth mask.

        Inputs:
            pred: Tensor [H, W]
            target: Tensor [H, W]
    """
    loss = torch.nn.functional.binary_cross_entropy(pred, target)
    # scale loss to [0, 1]
    loss = loss / 100
    return loss

def weighted_binary_CE(pred, target):
    """
        Compute the loss between a prediction mask and a ground truth mask.

        Inputs:
            pred: Tensor [H, W]
            target: Tensor [H, W]
    """
    # define weight as avg number of black pixels / avg number of all pixels
    all_pixels = target.shape[0]*target.shape[1]
    black_pixels = all_pixels - target.sum()
    weight = black_pixels / all_pixels
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, weight=weight)
    return loss

def dice_loss(pred, target):
    """
        Compute the loss between a prediction mask and a ground truth mask.

        Inputs:
            pred: Tensor [H, W]
            target: Tensor [H, W]
    """
    # compute dice loss
    intersection = (pred.int() & target.int()).float().sum((0,1))
    union = (pred.int() | target.int()).float().sum((0,1))
    loss = 1 - (2*intersection + 1e-6) / (union + 1e-6)
    return loss

def dice_bce_loss(pred, target, smooth=1):
    """
        Compute the loss between a prediction mask and a ground truth mask.

        Inputs:
            pred: Tensor [H, W]
            target: Tensor [H, W]
    """

    # flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # compute dice loss
    intersection = (pred * target).sum()
    dice_loss = 1 - (2.*intersection + smooth) / (pred.sum() + target.sum() + smooth)
    BCE = torch.nn.functional.binary_cross_entropy(pred, target, reduction='mean')
    Dice_BCE = BCE + dice_loss

    
    return BCE, dice_loss, Dice_BCE




def test():
    pred = torch.zeros((28, 28))
    target0 = torch.zeros((14, 7))
    target1 = torch.ones((14, 7))
    target_top = torch.cat((target0, target0, target1, target1), dim=1)
    target_bottom = torch.cat((target0, target0, target0, target0), dim=1)
    target = torch.cat((target_top, target_bottom), dim=0)

    target2 = target1.clone()
    target2[:, 0:50] = 0

    pred_top = torch.cat((target0, target0, target2, target1), dim=1)
    pred_bottom = torch.cat((target0, target0, target0, target0), dim=1)
    pred = torch.cat((pred_top, pred_bottom), dim=0)

    # pred = torch.rand((256, 256))
    # pred = target
    

    # test binary_CE
    loss = binary_CE(pred, target)
    print("BCE loss: {}".format(loss))

    # test weighted_binary_CE
    # pred = 255*torch.zeros((256, 256))
    # target = 255*target
    # pred = target
    loss = weighted_binary_CE(pred, target)
    print("Weighted BCE loss: {}".format(loss))

    # test dice_loss
    loss = dice_loss(pred, target)
    print("Dice loss: {}".format(loss))

    # test dice_bce_loss
    bce, dice, both = dice_bce_loss(pred, target)
    print("BCE2 loss: {}".format(bce))
    print("Dice2 loss: {}".format(dice))
    print("Dice BCE loss: {}".format(both))

    return 0

from typing import List, Tuple

import torch
from torch import Tensor


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def test_resnet():
    cfg = OmegaConf.load("3DAmodal/configs/config.yaml")

    data_root = "/Midgard/Data/tibbe/datasets/KINS/"
    dataset = get_obj_from_str(cfg.DATASET.NAME)(data_root)
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=1, partition="image", distributed=False)
    img, anns = next(iter(dataloader))
    # img = torch.rand((1, 3, 256, 832))
    img_lst = ImageList(img, [(256, 832)])

    resnet = resnet50(pretrained=True)
    FRCNN = fasterrcnn_resnet50_fpn(pretrained=True)
    FRCNN.eval()
    named_layers = dict(FRCNN.named_modules())
    FRCNN.backbone.body.register_forward_hook(get_activation('body'))
    FRCNN.backbone.fpn.register_forward_hook(get_activation('fpn'))
    FRCNN.rpn.register_forward_hook(get_activation('rpn'))
    FRCNN.roi_heads.box_roi_pool.register_forward_hook(get_activation('box_pooler'))
    FRCNN.roi_heads.register_forward_hook(get_activation('roi_heads'))
    out = FRCNN(img)
    resnet.eval()

    res1 = nn.Sequential(*list(resnet.children())[:5])
    res2 = nn.Sequential(*list(resnet.children())[5:6])
    res3 = nn.Sequential(*list(resnet.children())[6:7])
    res4 = nn.Sequential(*list(resnet.children())[7:8])

    features = {}

    features['0'] = res1(img)
    features['1'] = res2(features['0'])
    features['2'] = res3(features['1'])
    features['3'] = res4(features['2'])

    
    state_dict = torch.load("/Midgard/home/tibbe/3damodal/3DAmodal/checkpoint_orig/aisformer_r50_kins.pth", map_location=torch.device('cpu'))
    resnet_dict = resnet.state_dict()
    frcnn_state_dict = FRCNN.state_dict()
    # fpn = feature_pyramid_network([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4], 256, 256)
    fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256, extra_blocks=LastLevelMaxPool())
    anchor_gen = anchor_utils.AnchorGenerator(sizes=((256, 256, 256, 256)), aspect_ratios=((4, 8, 16, 32)))
    rpn_head = rpn.RPNHead(256, 4)
    region_prop = rpn.RegionProposalNetwork(anchor_generator=anchor_gen, head=rpn_head,
                                            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                                            batch_size_per_image=16, positive_fraction=0.5,
                                            pre_nms_top_n=dict(training=2000, testing=1000),
                                            post_nms_top_n=dict(training=2000, testing=1000),
                                            nms_thresh=0.7)
    region_prop.eval()
    rpn_FRCNN = FRCNN.rpn
    rpn_FRCNN.anchor_generator.sizes = rpn_FRCNN.anchor_generator.sizes[:-1]
    rpn_FRCNN.anchor_generator.aspect_ratios = rpn_FRCNN.anchor_generator.aspect_ratios[:-1]
    rpn_FRCNN.anchor_generator.cell_anchors = rpn_FRCNN.anchor_generator.cell_anchors[:-1]
    roi_head = FRCNN.roi_heads
    multi_roi = MultiScaleRoIAlign(['0', '1', '2', '3'], 14, 2)
    aisformer = AISFormer_orig(['cpu', 'cpu'], cfg)


    # resnet.load_state_dict(state_dict, strict=False)

    

    # for name,layer in resnet.named_children():
    #     print(f"resnet.{name}")
    #     if img.shape[1] == 256:
    #         break
    #     img = layer(img)
    #     print(img.shape)

    
    # named_layers = dict(FRCNN.named_modules())
    # modules_list = dict(resnet.named_modules())
    # modules_list.pop('fc')
    # modules_list.pop('avgpool')
    # modules_list.pop('')
    # modules_list = list(modules_list.values())

    # FRCNN.backbone.body = nn.Sequential(*modules_list)
    named_layers = dict(FRCNN.named_modules())

    # FRCNN.backbone.body.layer1.register_forward_hook(get_activation('layer1'))
    # FRCNN.backbone.body.layer2.register_forward_hook(get_activation('layer2'))
    # FRCNN.backbone.body.layer3.register_forward_hook(get_activation('layer3'))
    # FRCNN.backbone.body.layer4.register_forward_hook(get_activation('layer4'))
    
    resnet.layer1.register_forward_hook(get_activation('layer1'))
    resnet.layer2.register_forward_hook(get_activation('layer2'))
    resnet.layer3.register_forward_hook(get_activation('layer3'))
    resnet.layer4.register_forward_hook(get_activation('layer4'))
    
    out_res = resnet(img)
    input_feat = {'0': activation['layer1'], '1': activation['layer2'], '2': activation['layer3'], '3': activation['layer4']}
    out_fpn = fpn(input_feat)
    out_rpn = rpn_FRCNN(img_lst, out_fpn, None)
    # roi_head.box_roi_pool.output_size = (14, 14)
    # roi_head.box_head.fc6 = nn.Linear(256*14*14, 1024, bias=True)
    print(roi_head.box_roi_pool.output_size)
    out_roi = roi_head(out_fpn, out_rpn[0], img_lst.image_sizes)
    # out_roi = multi_roi(out_fpn, out_rpn[0], img_lst.image_sizes)
    # get bbs and apply multi roi
    bbs = out[0]['boxes'][torch.where(out[0]['scores'] > 0.8)]
    out_mask_roi = multi_roi(out_fpn, [bbs], img_lst.image_sizes)

    # AISformer
    out_aisformer = aisformer(out_mask_roi)


    # draw bbs on image
    



    
    # print(activation['rpn.head.conv'].shape)
    for name, layer in FRCNN.named_children():
        print(f"FRCNN.{name}")
        if name == "roi_heads":
            break
        img = layer(img)
        print(img.shape)
    

    print(img.shape) # [1, 3, 250, 828]
    # max val for each channel
    print(img.max(dim=2).values.max(dim=2).values)
    # min val for each channel
    print(img.min(dim=2).values.min(dim=2).values)
    # -> images are in range [0, 1]


    print()


if __name__ == '__main__':
    test_resnet()