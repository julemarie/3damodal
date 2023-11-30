import torch
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from pycocotools import mask as coco_mask
import cv2


def test_decode_mask():
    # Example RLE-encoded mask information
    mask_info = {"size": [1080, 1920],
                "counts": "^n\\n07<Nko02koNM:d0jo0@loNLOo0UP1`0I6L40000000000O3M3M3M2O1O000000000000000000O1OMjMfPOV2Yo0kMgPOS2[o0mMePOQ2]o0oMdPOo1\\o0RNdPOn1\\o0RNdPOn1\\o0RNdPOn1\\o0RNdPOn1[o0SNePOm1[o0SNePOm1[o0SNePOm1[o0TNbPOn1^o0RNbPOn1^o0RNbPOn1^o0RNbPOn1^o0RNbPOn1^o060kMbPOP2^o050kMbPOP2^o0PNbPOP2]o0QNcPOo1]o0QNcPOo1]o0QNcPOo1]o0QNcPOo1]o0QNcPOQ2[o0oMePOS2Yo0mMgPOT2Xo0lMhPOU2Wo0kMiPOV2Vo0iMjPOX2Vo0500OdMkPOW2Uo0iMlPOV2To0jMlPOU2Uo0kMkPOU2Uo0kMkPOU2Uo0kMkPOU2Uo0kMkPOT2Vo0lMjPOS2Wo0mMiPOR2Xo0nMhPOP2Zo0PNfPOP2Zo0701NhMiPOT2To0lMlPOV2Ro0jMnPOX2Po0hMPQOY2on0gMQQOY2on0gMQQOZ2nn0fMRQOZ2nn0fMQQOZ2Po0fMPQOZ2Po0fMPQOZ2Po0fMPQOZ2Po0fMPQOZ2on080^MRQOZ2nn0fMRQOZ2nn0fMRQOZ2nn0fMRQOZ2nn0fMRQOZ2nn0fMRQOZ2nn0fMRQOY2on0gMQQOY2on0gMQQOY2on0gMPQOY2Qo0gMoPOX2Qo0iMoPOT2To0kMmPOS2Uo0mMkPOS2Uo080eMkPOS2Uo0mMkPOS2Uo0mMkPOS2Uo0mMkPOS2Uo0mMkPOS2Uo0mMkPOS2Uo0mMkPOS2Uo0mMkPOS2Uo0mMkPOU2So0kMmPOW2Qo0iMoPOX2Po0gMQQOY2on0gMQQOZ2nn080^MRQO[2mn0eMSQO\\2ln0dMTQO\\2ln0dMTQO\\2ln0dMTQO\\2To00004K100000000000000N2O1O1O1M300000000000000000O10000001O002N0H^POZNlo0NPQO0Zfkk0"
                }

    # Decode RLE-encoded mask
    decoded_mask = coco_mask.decode(mask_info)

    print(type(decoded_mask))

    # print(decoded_mask[np.where(decoded_mask > 0)])

    # Convert to a binary mask
    # binary_mask = np.sum(decoded_mask, axis=2) > 0

    # Add channel dimension if needed
    binary_mask = torch.tensor(decoded_mask, dtype=torch.float).unsqueeze(0)

    print(binary_mask.shape)
    print(binary_mask[np.where(binary_mask > 0)])

    try:
        save_image(binary_mask, "test_mask.png")
    except:
        print("something went wrong")

    print("saved img")

def test_visible_mask():
    path = "/Midgard/Data/tibbe/datasets/AmodalSynthDrive/train/amodal_instance_seg/20230310230810_Clear_Sunset/front_full_0095_aminseg.png"

    img = cv2.imread(path)

    tf = transforms.ToTensor()

    img = tf(img)
    print(img.shape)

    try:
        save_image(img, "test_all.png")
        save_image(img[0], "test_0.png")
        save_image(img[1], "test_1.png")
        save_image(img[2], "test_2.png")
    except:
        print("something went wrong")

    print("saved imgs")

if __name__ == "__main__":
    test_visible_mask()