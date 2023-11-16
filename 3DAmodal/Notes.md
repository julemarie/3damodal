# Notes about Implementation

## AISFormer
- architecture is implemented according to paper but:
    - the code of AISFormer has some additional layers that are not mentioned in the paper (e.g. norm layers, "mask_feat_learner", or also different strucure of FFN than I have)
    - when i input an image from ASD, ROIAlign gives out the bboxes of instances but after Deconv, it's all one color --> because not trained yet?