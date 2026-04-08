# Assignment 4: Report artifacts
## Task 1: MNIST experiment with VAE - Reconstruction loss comparison (L1 vs L2)
1. Tune all hyperparameters with L2 reconstruction loss: already done this part, just record/point me to the results.
2. Keeping hyperparameters constant, switch reconstruction loss to L1: create a separate config file for same parameters but switch loss to L1. output dir to a separate folder in outputs and record/point me to the results
3. Show examples of L1 and L2 reconstructions (not generated samples): point me to the files
4. Show examples of L1 and L2 generated samples (using generate_images.py):
``` bash
cd assignment4; mamba activate cs7643-a4; python generate_images.py --config_file configs/config_vae_mnist_l1.yaml --output_dir outputs/vae_mnist_l1/images
cd assignment4; mamba activate cs7643-a4; python generate_images.py --config_file configs/config_vae_mnist.yaml --output_dir outputs/vae_mnist/images
```

## Task 2: MNIST experiment with VAE - tune parameters for few-shot KNN classification
Use configs/config_vae.yaml to tune for KNN classification.
Current remote tests (not available locally):
```
1.2.2) test_10_shot_knn_medium (test_VAE_KNN.TestVAEKNN) (0/1)
Test Failed: 0.3865 not greater than or equal to 0.4
1.2.3) test_10_shot_knn_hard (test_VAE_KNN.TestVAEKNN) (0/1)
Test Failed: 0.4206 not greater than or equal to 0.5
1.2.4) test_10_shot_knn_expert (test_VAE_KNN.TestVAEKNN) (0/0.75)
Test Failed: 0.3835 not greater than or equal to 0.6
```

## Task 3: GAN with MNIST - Mode collapse and hyperparameter tuning
There is a common failure mode of GAN training termed mode collapse. that is when the generator learns to fool the discriminator producing only a certain subset of P(x). Find hyperparameters that exhibit this failure mode and show generated examples. List your hyperparameters that obtained this.
Use configs/config_gan_mode_collapse.yaml to tune for GAN training. Output dir `outputs/gan_mnist_mode_collapse` for the mode collapse examples.

## Task 4: Diffusion models with MNIST
- Forward diffusion visual (any epoch)
- Reverse diffusion visual (epoch 0)
- Reverse diffusion visual (epoch 9)

## Task 5: All models with FashionMNIST
- Tune all models for good reconstructions and generations on FashionMNIST. (done)
- Generate grid images (using generate_images.py) for all models and calculate FID scores for all models.
``` bash
cd assignment4; mamba activate cs7643-a4;
python generate_images.py --config_file configs/config_<model>_fashion.yaml --output_dir outputs/<model>_fashion
python -m pytorch_fid outputs/<model>_fashion/images tests/assets/fid-stats-fashion.npz --device cuda
```