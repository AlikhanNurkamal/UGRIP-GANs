import os
import torch
import nibabel
import numpy as np
import gzip
from torch.utils.data import Dataset


class LabelsDataset(Dataset):
    def __init__(self, data_dir: str, crop_labels: bool = True, target_shape: tuple = (64, 64, 64), transforms = None):
        """_summary_

        Args:
            data_dir (str): directory containing the label files
            crop_labels (bool, optional): whether to crop 3d images to a specified target shape. Defaults to True.
            target_shape (tuple, optional): shape of the voxel that will be returned after cropping (and padding). Defaults to (64, 64, 64).
            transforms (Any, optional): transformations applied to the labels. Defaults to None.
        """
        self.patches_dir = data_dir + "train/patches/"
        self.labels_dir = data_dir + "train/seg_labels/"
        # self.crop_labels = crop_labels
        # self.target_shape = target_shape
        self.transforms = transforms
    
    # def _get_data(self, nifty, dtype="int16"):
    #     if dtype == "int16":
    #         data = np.abs(nifty.get_fdata().astype(np.int16))
    #         data[data == -32768] = 0
    #         return data
    #     return nifty.get_fdata().astype(np.uint8)
    
    # def _load_all_modalities(self, directory, example_id):
    #     all_modality = nibabel.load(os.path.join(directory, example_id, f"{example_id}-all_modalities.nii.gz"))
    #     affine, header = all_modality.affine, all_modality.header
        
    #     all_modality = self._get_data(all_modality, "int16")
    #     all_modality = nibabel.nifti1.Nifti1Image(all_modality, affine, header=header)
        
    #     return all_modality
    
    # def _load_seg_label(self, directory, example_id):
    #     seg = nibabel.load(os.path.join(directory, example_id, f"{example_id}-seg.nii.gz"))
    #     affine, header = seg.affine, seg.header
        
    #     seg = self._get_data(seg, "unit8")
    #     seg = nibabel.nifti1.Nifti1Image(seg, affine, header=header)
        
    #     return seg
    
    def __len__(self):
        return len(os.listdir(self.labels_dir))  # was self.data_dir
    
    def __getitem__(self, index):
        # example_id = os.listdir(self.data_dir)[index]
        
        # # Load the image and reshape it to have 1 channel
        # image = self._load_all_modalities(self.data_dir, example_id).get_fdata().astype(np.int16)
        # reshaped_image = np.max(image, axis=0)
        
        # # Load the label and reshape it to have 5 channels (5th channel is the brain mask)
        # label = self._load_seg_label(self.data_dir, example_id).get_fdata().astype(np.uint8)
        # reshaped_mask = np.zeros((5, 240, 240, 155))
        # for i in range(4):
        #     reshaped_mask[i] = (label == i).astype(int)
        # reshaped_mask[4] = (reshaped_image > 0).astype(int)
        
        # if self.transforms is not None:
        #     label = self.transforms(reshaped_mask)
        # else:
        #     label = torch.from_numpy(reshaped_mask)
        
        # label = label.float()
        # return label
        
        label_id = os.listdir(self.labels_dir)[index]
        label_file = gzip.GzipFile(os.path.join(self.labels_dir, label_id), "r")
        label = np.load(label_file)
        label_file.close()
        
        patch_id = label_id.replace("label", "img")
        patch_file = gzip.GzipFile(os.path.join(self.patches_dir, patch_id), "r")
        patch = np.load(patch_file)
        patch_file.close()
        
        reshaped_image = np.max(patch, axis=0)
        reshaped_mask = np.zeros((4, 128, 128, 128))
        reshaped_mask[0] = (reshaped_image > 0).astype(int)
        reshaped_mask[1] = label[0]
        reshaped_mask[2] = label[1]
        reshaped_mask[3] = label[2]
        
        if self.transforms is not None:
            label = self.transforms(reshaped_mask)
        else:
            label = torch.from_numpy(reshaped_mask)
        
        label = label.float()
        return label
