import torch
import numpy as np
import tifffile as tiff
from PIL import Image
import segmentation_models_pytorch as smp

def load_model(path):
    model = smp.Unet(encoder_name="mobilenet_v2", in_channels=11, classes=1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def preprocess_image(img_path):
    img = tiff.imread(img_path).astype(np.float32)
    norm_img = np.empty_like(img)
    for ch in range(img.shape[-1]):
        ch_data = img[:, :, ch]
        ch_min = ch_data.min()
        ch_max = ch_data.max()
        if ch_max - ch_min != 0:
            norm_img[:, :, ch] = (ch_data - ch_min) / (ch_max - ch_min)
        else:
            norm_img[:, :, ch] = 0
    red = norm_img[...,3]
    nir = norm_img[...,4]
    green = norm_img[...,2]

    # Generate new channels
    # NDWI — Normalized Difference Water Index
    ndwi = (green - nir) / (green + nir + 1e-9)

    # NDVI — Vegetation Index (can help distinguish land from water)
    ndvi = (nir - red) / (nir + red + 1e-9)

    # Expand Dimensions to add as channels
    ndwi = np.expand_dims(ndwi, axis=-1)
    ndvi = np.expand_dims(ndvi, axis=-1)

    # Add the new generated channels to the images
    feat_img = np.concatenate([norm_img, ndwi, ndvi], axis=-1)

    # Remove the least important channels (Blue [1], Green [2], Red [3])
    channels_to_remove = [1,2,3]
    feat_img = np.delete(feat_img, obj=channels_to_remove, axis=-1)

    tensor_img = torch.tensor(feat_img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor_img

def predict_mask(model, img_path, mask_save_path, input_rgb_path=None):
    img_tensor = preprocess_image(img_path)
    raw = tiff.imread(img_path).astype(np.float32)

    if input_rgb_path is not None:
        # Use NIR=4, SWIR1=5, SWIR2=6
        rgb_channels = [4, 5, 6]
        rgb_image = np.stack([raw[:, :, ch] for ch in rgb_channels], axis=-1)
        for i in range(3):
            ch = rgb_image[:, :, i]
            ch = 255 * (ch - ch.min()) / (ch.max() - ch.min()) if ch.max() != ch.min() else ch
            rgb_image[:, :, i] = ch
        rgb_image = rgb_image.astype(np.uint8)
        Image.fromarray(rgb_image).save(input_rgb_path)

    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        Image.fromarray(binary_mask).save(mask_save_path)