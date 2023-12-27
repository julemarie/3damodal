# Notes about Implementation

## AISFormer
- architecture is implemented according to paper but:
    - the code of AISFormer has some additional layers that are not mentioned in the paper (e.g. norm layers, "mask_feat_learner", or also different strucure of FFN than I have)
    - when i input an image from ASD, ROIAlign gives out the bboxes of instances but after Deconv, it's all one color --> because not trained yet?

Since the paper doesn't specify the dimensions of embedded images etc, I have to make assumptions and I decided to add a step that patchifies the ROIs from [K, C, H_m, W_M] to [K, H_m//P * W_m//P, C*P**2]. For me this makes more sense, because then we have a defined sequence lngth and embed dimension. Before this wasn't clear to me.


TODO:
- weight init
- double check model implementation with github
