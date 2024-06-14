import numpy as np
import os
import nibabel as nib
import gzip
import torchio as tio

#########################################
############# Initializations ###########
#########################################
# to be used from config file
image_size = [240, 240, 155]  # was [312, 384, 127]
patch_size = [128, 128, 128]
transform = tio.CropOrPad(target_shape=patch_size, padding_mode='constant')
num_patches = 20
datasets = ['train']
data_folder = '/home/alikhan.nurkamal/brats-project/large-dataset/'
target_folder = '/home/alikhan.nurkamal/brats-project/large-dataset-patches/'
str_patch_size = str(patch_size[0]) + 'x' + str(patch_size[1]) + 'x' \
                + str(patch_size[2])
save_folder = os.path.join(target_folder, str(num_patches) + '_ppp_'
                           + str_patch_size)

image_dtype = 'int16'  # was float32
label_dtype = 'uint8'

########################################
########################################

if not os.path.exists(save_folder):
    for dataset in datasets:
        os.makedirs(os.path.join(save_folder, dataset, 'patches'))
        os.makedirs(os.path.join(save_folder, dataset, 'seg_labels'))

# For systematic extraction finding the offset points
offsets = []
for i in range(len(image_size)):
    quotient = 0
    quotient, remainder = divmod(image_size[i], patch_size[i])
    # print(quotient, remainder)
    if remainder != 0:
        quotient += 1
    offset = quotient * patch_size[i] - image_size[i]
    # print(quotient, offset, remainder)
    if remainder != 0 and offset != 1: 
        if (offset % remainder) == 0:
            offset = offset//3
        elif (remainder % offset) == 0:
            offset = offset//4
        elif (offset > (patch_size[i]//2)) and (offset < (image_size[i]//2)):
            offset = offset//2
    offsets.append((quotient, offset))
# print(offsets)
xn, sys_offset_x = offsets[0]
yn, sys_offset_y = offsets[1]
zn, sys_offset_z = offsets[2]

# Extraction loop starts here
for dataset in datasets:
    print('Extracting for ', dataset)
    patient_folders = os.listdir(data_folder)  # was os.listdir(os.path.join(data_folder, dataset))
    for patient in sorted(patient_folders):
        num_extracted = 0
        print('Patient: ', patient)
        img_path = os.path.join(data_folder, patient, f'{patient}-all_modalities.nii.gz')  # was os.path.join(data_folder, dataset, patient, img_name)
        label_path = os.path.join(data_folder, patient, f'{patient}-seg.nii.gz')  # was os.path.join(data_folder, dataset, patient, label_name)
        print('---> Loading image...')
        img_nif = nib.load(img_path)
        image = np.array(img_nif.get_fdata(), dtype=image_dtype)
        print('---> Loading segmentation label...')
        seg_label_nif = nib.load(label_path)
        seg_label = np.array(seg_label_nif.get_fdata(), dtype=label_dtype)
        seg_label = np.expand_dims(seg_label, axis=0)
        print('Patient ', patient, ': image and label loaded.')
        print("Shape of image: ", image.shape, " and label: ", seg_label.shape)

        # -------------------------------------------------------------------
        # EXTRACT PATCHES SYSTEMATICALLY TO COVER THE ALL PARTS OF THE IMAGE
        # -------------------------------------------------------------------
        start_cx = 0
        for x in range(xn):
            stop_cx = start_cx + patch_size[0]
            start_cy = 0
            for y in range(yn):
                stop_cy = start_cy + patch_size[1]
                start_cz = 0
                for z in range(zn):
                    stop_cz = start_cz + patch_size[2]
                    
                    print(start_cx, stop_cx, start_cy, stop_cy, start_cz,
                          stop_cz)
                    patch = image[:, start_cx:stop_cx, start_cy:stop_cy,
                                  start_cz:stop_cz]
                    label = seg_label[:, start_cx:stop_cx, start_cy:stop_cy,
                                      start_cz:stop_cz]
                    
                    patch = transform(patch)
                    label = transform(label)
                    
                    # Reshape the label to have 3 channels
                    label = np.squeeze(label, axis=0)
                    reshaped_label = np.zeros((3, 128, 128, 128), dtype=np.uint8)
                    reshaped_label[0] = np.where((label == 2) | (label == 3), 1, 0)  # tumor core
                    reshaped_label[1] = np.where((label == 1) | (label == 2) | (label == 3), 1, 0)  # whole tumor
                    reshaped_label[2] = np.where(label == 2, 1, 0)  # enhancing tumor
                    label = reshaped_label
                    
                    print("Patch and label shape: ", patch.shape, " and ", label.shape)
                    start_cz = stop_cz - sys_offset_z
                    num_extracted += 1
                    
                    # save extracted patches as numpy arrays
                    img_patch_f = gzip.GzipFile(os.path.join(save_folder,
                                                             dataset,
                                                             'patches',
                                                             patient
                                                             +
                                                             '_img_systematic_'
                                                             +
                                                             str(num_extracted)
                                                             +
                                                             '.npy.gz'), 'w')
                    np.save(img_patch_f, np.asarray(patch))
                    img_patch_f.close()
                    
                    label_patch_f = gzip.GzipFile(os.path.join(save_folder,
                                                               dataset,
                                                               'seg_labels',
                                                               patient
                                                               +
                                                               '_label_systematic_'
                                                               +
                                                               str(num_extracted)
                                                               +
                                                               '.npy.gz'), 'w')
                    np.save(label_patch_f, np.asarray(label))
                    label_patch_f.close()

                start_cy = stop_cy - sys_offset_y
            start_cx = stop_cx - sys_offset_x
        print("Number of patches extracted systematically: ", num_extracted)
        
        # -----------------------------------------------------------
        # EXTRACT RANDOM PATCHES WITH TUMOUR IN THE CENTER OF EACH PATCH
        # -----------------------------------------------------------
        
        patches_to_be_extracted = num_patches - num_extracted
        random_extracted = 0
        print("Number of patches to be extracted randomly: ",
              patches_to_be_extracted)

        max_x, max_y, max_z = image_size
        min_x, min_y, min_z = 0, 0, 0

        # All voxel indices that is a tumour
        inds = np.asarray(np.where(seg_label > 0))  # was np.asarray(np.where(seg_label == 1))

        random_inds = inds[:, np.random.choice(inds.shape[1],
                                               patches_to_be_extracted,
                                               replace=False)]
        
        for i in range(patches_to_be_extracted):
            # get the coordinates of the random tumour around which the patch
            # will be extracted
            x = random_inds[1][i]  # was random_inds[0][i]
            y = random_inds[2][i]  # was random_inds[1][i]
            z = random_inds[3][i]  # was random_inds[2][i]

            random_img_patch = np.zeros(shape=([4] + patch_size), dtype=image_dtype)
            random_label_patch = np.zeros(shape=([1] + patch_size), dtype=label_dtype)
            
            # find the starting and ending x and y coordinates of given patch
            img_patch_start_x = max([x - int(patch_size[0]/2), min_x])
            img_patch_end_x = min([x + int(patch_size[0]/2), max_x])
            img_patch_start_y = max([y - int(patch_size[1]/2), min_y])
            img_patch_end_y = min([y + int(patch_size[1]/2), max_y])
            img_patch_start_z = max([z - int(patch_size[2]/2), min_z])
            img_patch_end_z = min([z + int(patch_size[2]/2), max_z])
            
            print(img_patch_start_x, img_patch_end_x, img_patch_start_y,
                  img_patch_end_y, img_patch_start_z, img_patch_end_z)
            
            offset_x = min_x - (x - int(patch_size[0]/2)) if x - int(patch_size[0]/2) < min_x else 0
            offset_y = min_y - (y - int(patch_size[1]/2)) if y - int(patch_size[1]/2) < min_y else 0
            offset_z = min_z - (z - int(patch_size[2]/2)) if z - int(patch_size[2]/2) < min_z else 0
            
            random_img_patch[:, offset_x : offset_x + (img_patch_end_x-img_patch_start_x),
                                offset_y : offset_y + (img_patch_end_y-img_patch_start_y),
                                offset_z : offset_z + (img_patch_end_z-img_patch_start_z)] \
                = image[:, img_patch_start_x:img_patch_end_x, img_patch_start_y:img_patch_end_y, img_patch_start_z:img_patch_end_z]
            
            random_label_patch[:, offset_x : offset_x + (img_patch_end_x-img_patch_start_x),
                                offset_y : offset_y + (img_patch_end_y-img_patch_start_y),
                                offset_z : offset_z + (img_patch_end_z-img_patch_start_z)] \
                = seg_label[:, img_patch_start_x:img_patch_end_x, img_patch_start_y:img_patch_end_y, img_patch_start_z:img_patch_end_z]
            
            # Reshape the label to have 3 channels
            random_label_patch = np.squeeze(random_label_patch, axis=0)
            reshaped_mask = np.zeros((3, 128, 128, 128), dtype=np.uint8)
            reshaped_mask[0] = np.where((random_label_patch == 2) | (random_label_patch == 3), 1, 0)  # tumor core
            reshaped_mask[1] = np.where((random_label_patch == 1) | (random_label_patch == 2) | (random_label_patch == 3), 1, 0)  # whole tumor
            reshaped_mask[2] = np.where(random_label_patch == 2, 1, 0)  # enhancing tumor
            random_label_patch = reshaped_mask
            
            print("Patch and label shape: ", random_img_patch.shape, " and ", random_label_patch.shape)
            
            num_extracted += 1
            random_extracted += 1
            
            # save extracted patches as numpy arrays
            img_patch_f = gzip.GzipFile(os.path.join(save_folder, dataset,
                                                     'patches', patient
                                                     + '_img_random_'
                                                     + str(num_extracted)
                                                     + '.npy.gz'), 'w')
            np.save(img_patch_f, np.asarray(random_img_patch))
            img_patch_f.close()
            
            label_patch_f = gzip.GzipFile(os.path.join(save_folder, dataset,
                                                       'seg_labels', patient
                                                       + '_label_random_'
                                                       + str(num_extracted)
                                                       + '.npy.gz'), 'w')
            np.save(label_patch_f, np.asarray(random_label_patch))
            label_patch_f.close()

        print("Number of patches extracted randomly: ", random_extracted)
print('DONE')
