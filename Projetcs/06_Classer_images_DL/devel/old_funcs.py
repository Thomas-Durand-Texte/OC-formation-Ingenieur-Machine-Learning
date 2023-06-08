#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def __image_resize_crop__(image: tensor, target_shape: tuple) -> tensor:
    """Resize input tensor corresponding to image according to
       the desired shape

    Args:
        image (tensor): input image
        target_shape (tuple): desired shape

    Returns:
        tensor: resized_image
    """
    ht, wt = target_shape
    d, h, w = image.shape
    if h > w:
        resize = transforms.Resize((ht, int(ht*w/h+0.5)), antialias=True)
        resized = resize(image)
        crop = transforms.CenterCrop((ht, wt))
        cropped = crop(resized)
    else:
        resize = transforms.Resize((int(wt*h/w+0.5), wt), antialias=True)
        resized = resize(image)
        crop = transforms.CenterCrop((ht, wt))
        cropped = crop(resized)
    return cropped


# %% END OF FILE
###
