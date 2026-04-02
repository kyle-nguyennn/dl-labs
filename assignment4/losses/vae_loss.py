import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, beta=1, recon_loss='l2', reduction='mean', return_losses=False):
        super(VAELoss, self).__init__()
        self.beta = beta
        if recon_loss.lower() == 'l2':
            self.reconstruction_loss = nn.MSELoss(reduction=reduction)
        if recon_loss.lower() =='l1':
            self.reconstruction_loss = nn.L1Loss(reduction=reduction)
        if recon_loss.lower() == 'bce':
            self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.return_losses = return_losses
    def forward(self, reconstructed, original, mu, logvar):
        loss, loss_recon, loss_kl = None, None, None
        #############################################################################
        # TODO:                                                                     #
        #    1. call the reconstruction loss defined in init with the appropriate args
        #    2. compute KL distance loss         
        #    3. compute beta weighted loss    
        #    4. use the if statement below to inform you of the containers to store the loss    #
        #############################################################################
        # print(f"mu shape: {mu.shape}")
        # print(f"input shape: {original.shape}")
        loss_recon = self.reconstruction_loss(reconstructed, original) * 784 # don't average per dimension
        var = torch.exp(logvar)
        loss_kl = -0.5*torch.sum((1 + logvar - mu**2 - var)) # sum along latent dim, and batch
        loss = loss_recon + self.beta*loss_kl
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        
        if self.return_losses:
            return loss, loss_recon, loss_kl
        else:
            return loss