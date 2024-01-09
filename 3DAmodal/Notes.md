# Notes about Implementation

## AISFormer
- architecture is implemented according to paper but:
    - the code of AISFormer has some additional layers that are not mentioned in the paper (e.g. norm layers, "mask_feat_learner", or also different strucure of FFN than I have)
    - when i input an image from ASD, ROIAlign gives out the bboxes of instances but after Deconv, it's all one color --> because not trained yet?

Since the paper doesn't specify the dimensions of embedded images etc, I have to make assumptions and I decided to add a step that patchifies the ROIs from [K, C, H_m, W_M] to [K, H_m//P * W_m//P, C*P**2]. For me this makes more sense, because then we have a defined sequence lngth and embed dimension. Before this wasn't clear to me.


What made it difficult to implement AISFormer?
- insufficient description of the architecture (e.g. dimension of C in model figure)
- description in paper does not match code implementation (conv layers switched)
- no description how loss is calculated wrt the occluder masks (there is no GT for that in KINS)
- vram constraints made it difficult to run same model size (using smaller version now)
- missing details:
    - number of encoder/decoder layers
    - dimension of C
    - 