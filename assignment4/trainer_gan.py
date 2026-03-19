from models.GAN import BasicDiscriminator, BasicGenerator, BasicLeakyGenerator
from utils.data_utils import set_seed, get_device, AverageMeter
from utils.trainer import Trainer
import os, torch, time, argparse
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml


class GANTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)

        self.generator = self._init_generator()
        self.discriminator = self._init_discriminator()

        self.generator.to(device=self.device)
        self.discriminator.to(device=self.device)
        self.optimizer_gen = self._init_optimizer(self.generator)
        self.optimizer_disc = self._init_optimizer(self.discriminator)
        self.criterion = torch.nn.BCELoss(reduction='mean')

        self.fixed_eval_latents = torch.randn((64, self.config.network.latent_dim), device=self.device)

 

    @staticmethod
    def _build_generator(output_dim, hidden_dim, latent_dim, leaky=False):
        """
        build basic generator given inputs
        """
        if leaky:
            return BasicLeakyGenerator(output_dim=output_dim, 
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim)
        else:
            return BasicGenerator(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  latent_dim=latent_dim)

    def _init_generator(self):
        """
        instantiate generator
        """
        generator = self._build_generator(output_dim=self.input_dim, 
                              hidden_dim=self.config.network.hidden_dim, 
                              latent_dim=self.config.network.latent_dim,
                              leaky=self.config.gan.leaky)
        return generator
    
    @staticmethod
    def _build_discriminator(input_dim, hidden_dim, leaky=True):
        """
        build basic discriminator given inputs
        """
        return BasicDiscriminator(input_dim=input_dim, 
                   hidden_dim=hidden_dim,
                   output_dim=1,
                   leaky=leaky)

    def _init_discriminator(self):
        """
        instantiate discriminator
        """
        disc = self._build_discriminator(input_dim=self.input_dim, 
                              hidden_dim=self.config.network.hidden_dim, 
                              leaky=self.config.gan.leaky
                                )
        return disc


    def compute_loss_real(self, data, batch_size):
        #############################################################################
        # TODO:
        # 1. sample real data, put on device, and create 'real' labels,
        # 2. obtain discriminater assessment on real images, obtain loss on real images using self.criterion
        #############################################################################
        return None

    def compute_loss_fake(self, batch_size, latent_dim):
        #############################################################################
        # TODO:
        # 1. sample latents for fake images, create 'fake' labels. dont forget to detatch generator output from graph.
        # 2. obtain discriminater assessment on fake images, obtain loss on fake images using self.criterion
        #############################################################################

        return None

    def compute_loss_gen(self, batch_size, latent_dim):
        #############################################################################
        # TODO:
        #  1. use generator with new latents to obtain fakes. create labels
        #  2. get discriminator assessment on fakes and compute loss using self.criterion
        #############################################################################

        return None


    def train(self):
        start = time.time()


        loss_meter_gen = AverageMeter()
        loss_meter_fake = AverageMeter()
        loss_meter_real = AverageMeter()
       # loss_meter = AverageMeter()
        iter_meter = AverageMeter()

        hidden_dim = self.config.network.hidden_dim
        latent_dim = self.config.network.latent_dim
        loss_func = self.criterion

        for epoch in range(self.n_epochs):
           
            for i, (data, _ ) in enumerate(self.train_loader):
                start = time.time()
                batch_size = data.size(0)

                """
                    complete training loop for GAN.
                    1. take real data, flatten and move to device. create targets for real data.
                    2. obtain discriminator predictions on real data and use targets to compute real component of loss.
                    3. sample latent (N, batch_size) and obtain fake images from generator. dont forget to remove generator outputs from graph. create fake targets and use to compute fake images loss.
                    4. compute overall discriminator loss and backprop
                    5. sample latent z (N, batch_size) and use on generator.
                """
                loss_real = None
                loss_fake = None
                loss_gen = None
                #############################################################################
                # TODO:  
                # A. Train discriminator
                # 0. Implement compute_loss_real and compute_loss_fake
                # 1. Compute and combine the fake and real losses and take step to optimize discriminator.

                # B. Train generator
                # 0. Implement compute_loss_gen
                # 1. compute the generator loss and take step to optimize generator.
                #############################################################################


                #############################################################################
                #                              END OF YOUR CODE                             #
                ############################################################################# 


                loss_meter_real.update(loss_real, data.size(0))
                loss_meter_fake.update(loss_fake, data.size(0))
                loss_meter_gen.update(loss_gen, data.size(0))
                iter_meter.update(time.time()-start)

                if i % 1000 == 0:
                    print(
                        f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'D_Loss_Real {loss_meter_real.val:.3f} ({loss_meter_real.avg:.3f})\t',
                        f'D_Loss_Fake {loss_meter_fake.val:.3f} ({loss_meter_fake.avg:.3f})\t',
                        f'G_Loss {loss_meter_gen.val:.3f} ({loss_meter_gen.avg:.3f})\t',
                        f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
                        )

                if (epoch*len(self.train_loader)+i) % 3000 == 0:  
                    filename = f"{self.output_dir}/train_epoch_{epoch+1}_iter_{i}.png"               
                    self.sample_generator(self.fixed_eval_latents, filename, n=self.fixed_eval_latents.size(0))

        filename = f"{self.output_dir}/final_samples.png"               
        self.sample_generator(self.fixed_eval_latents, filename, n=self.fixed_eval_latents.size(0))
        print(f"Completed in {(time.time()-start):.3f}.")
        self.save_model(self.discriminator, f'{self.output_dir}/discriminator_{self.dataset.lower()}.pth')
        self.save_model(self.generator, f'{self.output_dir}/generator_{self.dataset.lower()}.pth')


    def sample_generator(self, data, filename, n):
        self.generator.eval()  
        with torch.no_grad():
            res = self.generator(data.to(self.device))
        vutils.save_image(res.view(-1,1,28,28), filename, nrow=n)
        self.generator.train()