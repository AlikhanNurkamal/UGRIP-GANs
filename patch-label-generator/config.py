##################################
# ####### Train settings ####### #
##################################
list_config_names = ['Trial', 'seed', 'ngpu', 'image_size', 'nz',
                     'batch_size', 'lrd', 'lrg', 'spectral_norm_D',
                     'spectral_norm_G', 'gp', 'nc', 'n_disc',
                     'beta1', 'beta2', 'num_workers', 'k_size', 'ngf', 'ndf',
                     'max_grad_norm', 'noise_m', 'clip_param']

trial_num = 1
continue_train = False  # if true, also update the continue train parameters
seed = 42

# Number of training epochs
num_epochs = 100

# Save model or not, how often to save
# and format to save samples
is_model_saved = True
save_n_epochs = 10
save_nifti = True  # If false saved as npy.gz

use_mixed_precision = True

# Number of workers for dataloader
workers = 8

# Spatial size of training images. All images need to be of the same size
image_size = (128, 128, 128)  # was (128, 128, 64)

num_images = 1251 * 20  # was 2350

################################
# ######## GPU settings ###### #
################################
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# gpu number
cuda_n = [0]

# Root directory for dataset
dataroot = ""

save_model = "./checkpoints/models/trial_" + str(trial_num) + "/"
save_results = "./checkpoints/results/trial_" + str(trial_num) + "/"
save_config = "./checkpoints/WGAN_GP_trials.csv"

####################################
# ###### parameter settings ###### #
####################################
spectral_norm_D = True  # False for GP model and DPGAN, True for all others
n_disc = 1  # 5 for DPGAN
spectral_norm_G = False
gp = True
# WGAN GP hyperparamter settings
lambdaa = 10

# Batch size during training
batch_size = 4

# kernel sizes
kd = 3
kg = 3

# Number of channels in the training images
nc = 7  # it is 7 because we have 4 modalities and 3 labels

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator ngf (max)
ngf = 512  # 1024 for c-SN-MP model, 256 for DPGAN and 512 for rest

# Size of feature maps in critic ndf (max)
ndf = 512  # 1024 for c-SN-MP model, 256 for DPGAN and 512 for rest 

# Learning rate for optimizers
lrd = 0.0004  # 0.0001 for DPGAN
lrg = 0.0002  # 0.0001 for DPGAN

# Beta hyperparameters for Adam optimizers
beta1d = 0
beta2d = 0.9
beta1g = 0
beta2g = 0.9

# Differential privacy parameters
max_norm_dp = 1  # clipping parameter for DP
noise_m = 0.3  # noise multiplier [experiments with 0.1, 0.3, 0.5]
alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
delta = 1 / num_images
secure_rng = True
add_clip = True  # additional weight clipping
clip_param_W = 0.01  # clipping parameter for WGAN

list_config = [trial_num, seed, ngpu, image_size, nz,
               batch_size, lrd, lrg, spectral_norm_D,
               spectral_norm_G, gp, nc,
               n_disc, beta1d, beta2d, workers, kd, ngf, ndf,
               max_norm_dp, noise_m, clip_param_W]

# ######################################################## #
# ######### Config params for continue training ########## #
# ######################################################## #

epoch_num_to_continue = 20
trial_num_to_continue = 2

saved_model_path = "./checkpoints/models/trial_" + \
                    str(trial_num_to_continue) + "/" \
                    + "epoch_" + str(epoch_num_to_continue) + ".pth"

# ######################################################## #
# ######### Config params for generate patches ########### #
# ######################################################## #
model_trial = 1
model_epoch = 99
test_batch_size = 2
gen_batch = 1
n_test_samples = 100  # was 11750
gen_threshold = 0.2  # in the case of SN model it is 0.2, for others it is 0.3
diff_priv = False


load_model_path = "./checkpoints/models/trial_" + str(model_trial) \
                  + "/" + "epoch_" + str(model_epoch) + ".pth"
gen_path = "./checkpoints/generated_images/trial_" + str(model_trial) \
           + "/" + "gen_images_epoch_" + str(model_epoch) \
            + "_threshold_" + str(gen_threshold) \
            + "_" + str(gen_batch) + "/train/for_segmentation/train/"
