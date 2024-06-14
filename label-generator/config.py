IMG_SHAPE = (128, 128, 128)

#_______________________________________________Constants_______________________________________________
DEBUG = True
PATH_DATASET = "/home/alikhan.nurkamal/brats-project/large-dataset/"

# Neural net
ARCHITECTURE = "arch2"
BATCH_SIZE = 5
WORKERS = 8
EPS = 1e-15  # for calc_gradient_penalty
TOTAL_ITER = 50000

# loss1 = -cd_z_hat_loss*CD_Z_HAT_WEIGHT - d_loss*D_WEIGHT + mse_loss*MSE_WEIGHT + gd_loss*GD_WEIGHT
CD_Z_HAT_WEIGHT = 1
D_WEIGHT = 1
MSE_WEIGHT = 100 
GD_WEIGHT = 1/100

#----------------------
# loss2 = x_loss2*X_LOSS2_WEIGHT + (gradient_penalty_r+gradient_penalty_h)*GP_D_WEIGHT
X_LOSS2_WEIGHT = 1
GP_D_WEIGHT = 100

#----------------------
# loss3 = x_loss3*X_LOSS3_WEIGHT + gradient_penalty_cd * GP_CD_WEIGHT
X_LOSS3_WEIGHT = 1
GP_CD_WEIGHT = 100

#----------------------
# setting latent variable sizes
LATENT_DIM = 500

# whether to save images and model checkpoints and their respective directories
SAVE_IMAGES = True  # save images in .nii.gz format
SAVE_MODELS = True
SAVE_IMAGES_DIR = "/home/alikhan.nurkamal/brats-project/gans-for-brats/label-generator/images/"
SAVE_MODELS_DIR = "/home/alikhan.nurkamal/brats-project/gans-for-brats/label-generator/checkpoints/"
