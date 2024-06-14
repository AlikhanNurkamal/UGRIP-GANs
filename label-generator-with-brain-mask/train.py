# Import operating system
import os

# To Show Images
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Import Pytorch
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Import Custom Dataset
from custom_dataset import LabelsDataset

# Monai (Import data and transform data)
from monai.transforms import \
    Compose, AddChannel, ScaleIntensity, ToTensor, Resize, RandRotate, RandFlip, RandScaleIntensity, RandZoom, RandGaussianNoise, RandAffine, ResizeWithPadOrCrop

# Import model and configuarations
from WGAN_SigmaRat2 import *
from config import *

architecture = "arch2"
DEVICE = torch.device('cuda:0')

# # Networks 
# ----
# ### Nomenclature 
# * G -> Generator
# * CD -> Code Discriminator
# * D -> Discriminator
# * E -> Encoder
def nets():
    G = Generator(noise=LATENT_DIM).to(DEVICE)
    CD = Code_Discriminator(code_size=LATENT_DIM, num_units=4096).to(DEVICE)
    D = Discriminator().to(DEVICE)
    
    E = Encoder(out_class=LATENT_DIM, is_dis=False).to(DEVICE)
    
    #______________________OPTIMIZERS______________________
    g_optimizer = optim.AdamW(G.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    d_optimizer = optim.AdamW(D.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    e_optimizer = optim.AdamW(E.parameters(), lr = 0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    cd_optimizer = optim.AdamW(CD.parameters(), lr = 0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    
    return (G, CD, D, E, g_optimizer, cd_optimizer, d_optimizer, e_optimizer)


G, CD, D, E, g_optimizer, cd_optimizer, d_optimizer, e_optimizer = nets()

if DEBUG:
    print("----------------------------------------------------Networks----------------------------------------------------")
    print("--------------------------Generator--------------------------")
    print(G)
    print("--------------------------Discriminator--------------------------")
    print(D)
    print("--------------------------Encoder--------------------------")
    print(E)
    print("--------------------------Code-Discriminator--------------------------")
    print(CD)

criterion_mse = nn.MSELoss()

Lloss1 = list()
Lloss2 = list()
Lloss3 = list()
MSELossL = list()
Gd_LossL = list()

# # Data Set Creator
# ----
# ### Using Monai Transformation
# * Resize (64x64x64)
# * Rotate (3º)
# * Flip (x axis)
# * Rand Scale Intensity (±0.1)
# * Zoom (1.1)
# * Gaussian Noise (0.01)
# * Scale Intensity Norm (-1,1)
# * Translate (4x4x0)
# * To Tensor
def create_train_loader():
    # train_transforms = Compose([RandFlip(prob=0.1, spatial_axis=0),
    #                             RandZoom(prob=0.1, min_zoom=(1.0), max_zoom=(1.1), mode="nearest"),
    #                             ToTensor()])
    
    train_ds = LabelsDataset(data_dir=PATH_DATASET)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=WORKERS, pin_memory=torch.cuda.is_available())
    return train_loader

train_loader = create_train_loader()


# # Gradient Penalty and Gradient Difference Loss
# ----
#___________________________________________WGAN-GP gradient penalty___________________________________________
def calc_gradient_penalty(model, x, x_gen):
    assert x.size() == x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha = torch.zeros(size=alpha_size, dtype=torch.float32, device=DEVICE).uniform_()
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat.requires_grad_(True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+EPS).sum(-1).sqrt()
    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty


'''
https://github.com/Y-P-Zhang/3D-GANs-pytorch/blob/master/models/losses.py
'''
#_____________________Gradient difference loss function_____________________
def gdloss(real,fake):
    dreal = real[:, :, 1:, 1:, 1:] - real[:, :, :-1, :-1, :-1] #[BCHWD]
    dfake = fake[:, :, 1:, 1:, 1:] - fake[:, :, :-1, :-1, :-1]
    gd_loss = torch.sum((torch.abs(dreal) - torch.abs(dfake))**2, dim=(0, 1, 2, 3, 4))

    return gd_loss


def networks_params(D, Dtruth, CD, CDtruth, E, Etruth, G, Gtruth):
    for p in D.parameters():  
        p.requires_grad = Dtruth
    for p in CD.parameters():  
        p.requires_grad = CDtruth
    for p in E.parameters():  
        p.requires_grad = Etruth
    for p in G.parameters():  
        p.requires_grad = Gtruth
        
    return (D, CD, E, G)


def visualization(image, img_name, iteration):
    image = image.detach().cpu().numpy()
    image = image[0]  # get the first image in the batch
    
    if img_name == "x_hat" or img_name == "x_rand":
        # Thresholding the brain segmentation mask
        image[0][image[0] > 0.5] = 1
        image[0][image[0] <= 0.5] = 0
        
        # If there is no brain mask, then there cannot be any tumor mask
        image[1:][image[0] == 0] = 0
        
        # Thresholding the tumor segmentation mask
        image[1:][image[1:] > 0.3] = 1
        image[1:][image[1:] <= 0.3] = 0
    
    # Sum all the channels
    image = torch.sum(image, dim=0)
    img = nib.nifti1.Nifti1Image(image, affine=np.eye(4))
    nib.save(img, f"{SAVE_IMAGES_DIR}" + f"{img_name}_iter{iteration}.nii.gz")


def load_dict(start):
    checkpoints = torch.load(SAVE_MODELS_DIR + f"checkpoint_{start}.pth")
    G.load_state_dict(checkpoints['G'])
    CD.load_state_dict(checkpoints['CD'])
    D.load_state_dict(checkpoints['D'])
    E.load_state_dict(checkpoints['E'])
    
    g_optimizer.load_state_dict(checkpoints['g_optimizer'])
    d_optimizer.load_state_dict(checkpoints['d_optimizer'])
    e_optimizer.load_state_dict(checkpoints['e_optimizer'])
    cd_optimizer.load_state_dict(checkpoints['cd_optimizer'])
    
    return (G, CD, D, E, g_optimizer, cd_optimizer, d_optimizer, e_optimizer)


# # Training
# ----
def training(resume, start):
    g_iter = 1
    d_iter = 1
    cd_iter = 1
    
    if resume:
        G, CD, D, E, g_optimizer, cd_optimizer, d_optimizer, e_optimizer = load_dict(start)
    else:
        G, CD, D, E, g_optimizer, cd_optimizer, d_optimizer, e_optimizer = nets()
    
    print('Starting training...')
    for iteration in range(start + 1, TOTAL_ITER + 1):
        ###############################################
        # Train Encoder - Generator 
        ###############################################
        D, CD, E, G = networks_params(D=D, Dtruth=False, CD=CD, CDtruth=False, E=E, Etruth=True, G=G, Gtruth=True)
        
        for iters in range(g_iter):
            G.zero_grad()
            E.zero_grad()
            real_images = next(iter(train_loader))  # next image
            _batch_size = real_images.size(0)
            
            real_images = real_images.to(DEVICE)  # next image into de GPU
            z_rand = torch.randn((_batch_size, LATENT_DIM)).to(DEVICE)  # random vector
            z_hat = E(real_images).view(_batch_size, -1)  # Output vector of the Encoder
            x_hat = G(z_hat)  # Generation of an image using the encoder's output vector
            x_rand = G(z_rand)  # Generation of an image using a random vector
            
            # Code discriminator absolute value of the encoder's output vector
            cd_z_hat_loss = CD(z_hat).mean()  # considering the random vector as real
            
            # Calculation of the discriminator loss
            d_real_loss = D(x_hat).mean()  # of the output vector of the Encoder
            d_fake_loss = D(x_rand).mean() # of the random vector
            d_loss = d_fake_loss + d_real_loss
            
            #_____________Mean_Squared_Error_____________
            mse_loss = criterion_mse(x_hat, real_images) 

            #_____________Gradient_Different_Loss_____________
            gd_loss = gdloss(real_images, x_hat).item()  # considering axis -> x,y,z

            #__________Generator_Loss__________
            loss1 = -cd_z_hat_loss*CD_Z_HAT_WEIGHT - d_loss*D_WEIGHT + mse_loss*MSE_WEIGHT + gd_loss*GD_WEIGHT

            if iters < g_iter-1:
                loss1.backward()
            else:
                loss1.backward(retain_graph=True)
            e_optimizer.step()
            g_optimizer.step()
            g_optimizer.step()

        ###############################################
        # Train Discriminator
        ###############################################
        D, CD, E, G = networks_params(D=D, Dtruth=True, CD=CD, CDtruth=False, E=E, Etruth=False, G=G, Gtruth=False)
        
        for iters in range(d_iter):
            d_optimizer.zero_grad()
            real_images = next(iter(train_loader))  # next image
            _batch_size = real_images.size(0)
             
            z_rand = torch.randn((_batch_size, LATENT_DIM)).to(DEVICE)  # random vector
            real_images = real_images.to(DEVICE)  # next image into de GPU
            z_hat = E(real_images).view(_batch_size, -1)  # Output vector of the Encoder
            x_hat = G(z_hat)  # Generation of an image using the encoder's output vector
            x_rand = G(z_rand)  # Generation of an image using a random vector
            
            #calculation of the discriminator loss (if it can distinguish between real and fake)
            x_loss2 = -2*D(real_images).mean() + D(x_hat).mean() + D(x_rand).mean() 
            
            #calculation of the gradient penalty
            gradient_penalty_r = calc_gradient_penalty(D,real_images.data, x_rand.data)
            gradient_penalty_h = calc_gradient_penalty(D,real_images.data, x_hat.data)
            
            #__________Discriminator_loss__________
            loss2 = x_loss2*X_LOSS2_WEIGHT + (gradient_penalty_r+gradient_penalty_h)*GP_D_WEIGHT
            loss2.backward(retain_graph=True)
            d_optimizer.step()

        ###############################################
        # Train Code Discriminator
        ###############################################
        D, CD, E, G = networks_params(D=D, Dtruth=False, CD=CD, CDtruth=True, E=E, Etruth=False, G=G, Gtruth=False)    
        
        for iters in range(cd_iter):
            cd_optimizer.zero_grad()
            
            #random vector (considered as real here)
            z_rand = torch.randn((_batch_size, LATENT_DIM)).to(DEVICE)
            
            #Gradient Penalty between randon vector and encoder's output vector
            gradient_penalty_cd = calc_gradient_penalty(CD, z_hat.data, z_rand.data) 
            
            x_loss3 = -CD(z_rand).mean() + CD(z_hat).mean()
            
            #___________Code_Discriminator_Loss___________
            loss3 = x_loss3*X_LOSS3_WEIGHT + gradient_penalty_cd * GP_CD_WEIGHT
            loss3.backward(retain_graph=True)
            cd_optimizer.step()

        ###############################################
        # Visualization
        ###############################################
        if iteration % 100 == 0:
            # Save losses in lists and print instant loss values
            Lloss1.append(float(loss1.data.cpu().numpy()))
            Lloss2.append(float(loss2.data.cpu().numpy()))
            Lloss3.append(float(loss3.data.cpu().numpy()))
            MSELossL.append(float(mse_loss.data.cpu().numpy()))
            Gd_LossL.append(float(gd_loss))
            
            fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 20))
            ax[0].plot(Lloss1)
            ax[0].set_title('Encoder-Generator Loss')
            
            ax[1].plot(Lloss2)
            ax[1].set_title('Discriminator Loss')
            
            ax[2].plot(Lloss3, label='Loss3')
            ax[2].set_title('Code Discriminator Loss')
            
            ax[3].plot(MSELossL, label='MSE_Loss')
            ax[3].set_title('MSE Loss')
            
            ax[4].plot(Gd_LossL, label='Gd_Loss')
            ax[4].set_title('Gradient Difference Loss')
            ax[4].set_xlabel('Iterations')
            plt.savefig('losses.png')
            
            print('[{}/{}]'.format(iteration,TOTAL_ITER),
                        f'D: {loss2.data.cpu().numpy():<8.3}', 
                        f'En_Ge: {loss1.data.cpu().numpy():<8.3}',
                        f'Code: {loss3.data.cpu().numpy():<8.3}',
                        f'MSE_Loss: {mse_loss.data.cpu().numpy():<8.3}',
                        f'Gd_Loss: {gd_loss:<8.3}',
                        )
            
            if SAVE_IMAGES:
                if not os.path.exists(SAVE_IMAGES_DIR):
                    os.makedirs(SAVE_IMAGES_DIR)
                # Show the real image
                visualization(real_images, img_name="Real", iteration=iteration)
                
                # Show the generated image using encoder's output vector
                visualization(x_hat, img_name="x_hat", iteration=iteration)
                
                # Show the generated image using a random vector
                visualization(x_rand, img_name="x_rand", iteration=iteration)
        
        ###############################################
        # Model Save
        ###############################################
        if iteration % 500 == 0:
            if SAVE_MODELS:
                checkpoints = {
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'E': E.state_dict(),
                    'CD': CD.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'e_optimizer': e_optimizer.state_dict(),
                    'cd_optimizer': cd_optimizer.state_dict(),
                    'iteration': iteration
                }
                
                if not os.path.exists(SAVE_MODELS_DIR):
                    os.makedirs(SAVE_MODELS_DIR)
                torch.save(checkpoints, SAVE_MODELS_DIR + f"checkpoint_{iteration}.pth")
        
        resume = True


def main():
    training(resume=True, start=10000)


if __name__ == "__main__":
    main()
