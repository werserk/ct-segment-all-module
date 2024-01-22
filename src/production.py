import torch
from .utils import preprocess_nifti, normalize, postprocess_output


def make_prediction(model, nifti):
    preprocessed = preprocess_nifti(nifti)
    normalized = normalize(preprocessed)
    with torch.no_grad():
        output = model(normalized)[0]
    array = postprocess_output(output)
    return array
