import torch
import numpy as np

def calc_center_bb(binary_class_mask):
    """Calculate the bounding box of the object in the binary class mask.
    Args:
        binary_class_mask - (batch_size x H x W): Binary mask isolating the hand.
    Returns:
        centers - (batch_size x 2): Center of mass calculation of the hand.
        bbs - (batch_size x 4): Bounding box of containing the hand. [x_min, y_min, x_max, y_max]
        crops - (batch_size x 2): Size of crop defined by the bounding box.
    """

    binary_class_mask = binary_class_mask.to(torch.int32)
    binary_class_mask = torch.eq(binary_class_mask, 1)
    if len(binary_class_mask.shape) == 4:
        binary_class_mask = binary_class_mask.squeeze(1)

    s = binary_class_mask.shape
    assert len(s) == 3, "binary_class_mask must be 3D."

    bbs = []
    centers = []
    crops = []

    for i in range(s[0]):
        if len(binary_class_mask[i].nonzero().shape) < 2:
            bb = torch.zeros(2, 2,
                             dtype=torch.int32,
                             device=binary_class_mask.device)
            bbs.append(bb)
            centers.append(torch.tensor([160, 160],
                                        dtype=torch.int32,
                                        device=binary_class_mask.device))
            crops.append(torch.tensor(100,
                                      dtype=torch.int32,
                                      device=binary_class_mask.device))
            continue
        else:
            y_min = binary_class_mask[i].nonzero()[:, 0].min().to(torch.int32)
            x_min = binary_class_mask[i].nonzero()[:, 1].min().to(torch.int32)
            y_max = binary_class_mask[i].nonzero()[:, 0].max().to(torch.int32)
            x_max = binary_class_mask[i].nonzero()[:, 1].max().to(torch.int32)

        start = torch.stack([y_min, x_min])
        end = torch.stack([y_max, x_max])
        bb = torch.stack([start, end], 1)
        bbs.append(bb)

        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        center = torch.stack([center_y, center_x])
        centers.append(center)

        crop_size_x = x_max - x_min
        crop_size_y = y_max - y_min
        crop_size = max(crop_size_y, crop_size_x)
        crops.append(crop_size)

    bbs = torch.stack(bbs)
    centers = torch.stack(centers)
    crops = torch.stack(crops)

    return centers, bbs, crops
