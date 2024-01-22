import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import torch
import json
from .constants import INFERENCE_SIZE

json_path = "data/classes.json"
with open(json_path) as f:
    json_data = json.load(f)


def load_nifti(filepath):
    survey = nib.load(filepath)
    survey = nib.Nifti1Image(survey.get_fdata(), survey.affine)
    survey = nib.as_closest_canonical(survey)
    survey = np.array(survey.dataobj, dtype=np.float32)
    return survey


def preprocess_nifti(survey):
    resized = zoom(
        survey, (
            INFERENCE_SIZE / survey.shape[0],
            INFERENCE_SIZE / survey.shape[1],
            INFERENCE_SIZE / survey.shape[2]
        )
    )
    resized = np.expand_dims(resized, 0)
    resized = np.expand_dims(resized, 0)
    resized = torch.from_numpy(resized)
    return resized


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def postprocess_output(output):
    array = output.detach().cpu().numpy()
    array = np.argmax(array, axis=0)
    return array


def get_mask_by_class(image, name):
    for j in json_data:
        if json_data[j] == name:
            res = int(j)
            return np.array(image == res, dtype=np.uint8)


def windowing(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img
