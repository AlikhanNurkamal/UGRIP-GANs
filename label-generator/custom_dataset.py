import os
import torch
import nibabel
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import find_objects


class LabelsDataset(Dataset):
    def __init__(self, data_dir: str, crop_labels: bool = True, target_shape: tuple = (64, 64, 64), transforms = None):
        """Dataset class for loading 3D tumor segmentation labels

        Args:
            data_dir (str): directory containing the label files.
            crop_labels (bool, optional): whether to crop 3d images to a specified target shape. Defaults to True.
            target_shape (tuple, optional): shape of the voxel that will be returned after cropping (and padding). Defaults to (64, 64, 64).
            transforms (Any, optional): transformations applied to the labels. Defaults to None.
        """
        self.data_dir = data_dir
        self.crop_labels = crop_labels
        self.target_shape = target_shape
        self.transforms = transforms
    
    def _get_data(self, nifty, dtype="int16"):
        if dtype == "int16":
            data = np.abs(nifty.get_fdata().astype(np.int16))
            data[data == -32768] = 0
            return data
        return nifty.get_fdata().astype(np.uint8)
    
    def _load_seg_label(self, directory, example_id):
        seg = nibabel.load(os.path.join(directory, example_id, f"{example_id}-seg.nii.gz"))
        affine, header = seg.affine, seg.header
        
        seg = self._get_data(seg, "unit8")
        seg = nibabel.nifti1.Nifti1Image(seg, affine, header=header)
        
        return seg
    
    def __len__(self):
        return len(os.listdir(self.data_dir))
    
    def __getitem__(self, index):
        example_id = os.listdir(self.data_dir)[index]
        
        label = self._load_seg_label(self.data_dir, example_id).get_fdata().astype(np.uint8)
        
        if self.crop_labels:
            # Find the bounding box of non-zero labels
            bounding_boxes = find_objects(label > 0)
            
            # Check if any non-zero labels exist
            if bounding_boxes:
                # Get the first bounding box
                bbox = bounding_boxes[0]
                
                label = label[bbox]
                
                if label.shape != self.target_shape:
                    # If the label shape is larger than the target shape, crop it more
                    label = label[:self.target_shape[0], :self.target_shape[1], :self.target_shape[2]]
                    
                    # If the label shape is smaller than the target shape, add padding
                    # Calculate padding sizes for each dimension
                    pad_sizes = []
                    for max_dim, cropped_dim in zip(self.target_shape, label.shape):
                        pad_before = (max_dim - cropped_dim) // 2
                        pad_after = max_dim - cropped_dim - pad_before
                        pad_sizes.append((pad_before, pad_after))
                    
                    label = np.pad(label, pad_sizes, mode="constant")
        
        # Reshape the mask to have 4 channels
        reshaped_mask = np.zeros((4, 64, 64, 64))
        for i in range(4):
            reshaped_mask[i] = (label == i).astype(int)
        
        if self.transforms is not None:
            label = self.transforms(reshaped_mask)
        else:
            label = torch.from_numpy(reshaped_mask)
        
        label = label.float()
        return label
