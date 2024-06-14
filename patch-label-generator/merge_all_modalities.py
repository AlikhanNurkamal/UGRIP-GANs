import nibabel
import os
import numpy as np


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def load_modalities_and_merge(directory, example_id, list_modalities=["t2f", "t1n", "t1c", "t2w"]):
    modalities = [
        nibabel.load(os.path.join(directory, example_id, f'{example_id}-{modality}.nii.gz'))
        for modality in list_modalities
    ]
    affine, header = modalities[0].affine, modalities[0].header
    
    vol = np.stack([get_data(modality, "int16") for modality in modalities], axis=0)
    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
    
    return vol


source_directory = "/home/alikhan.nurkamal/brats-project/large-dataset/"
example_ids = os.listdir(source_directory)

print("Start merging modalities...")
for idx, example_id in enumerate(example_ids):
    image = load_modalities_and_merge(source_directory, example_id)
    nibabel.save(image, os.path.join(source_directory, example_id, f'{example_id}-all_modalities.nii.gz'))
    if idx % 100 == 0:
        print(f"Processed {idx + 1} examples...")
print("Finished merging modalities!")
