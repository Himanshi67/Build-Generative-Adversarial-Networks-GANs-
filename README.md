# Build-Generative-Adversarial-Networks-GANs-
Build-Generative-Adversarial-Networks (GANs)
This project implements a Generative Adversarial Network (GAN) to generate images resembling those in the MNIST dataset. The GAN model consists of a Generator and a Discriminator, trained with PyTorch.

Table of Contents
Project Overview
Requirements
Model Architecture
Training the Model
Usage
Results
References
Project Overview
This project uses a GAN to generate synthetic images by learning the MNIST dataset's distribution. The goal of the model is to create images that resemble handwritten digits (0-9). The Generator creates fake images, while the Discriminator learns to distinguish between real and fake images.

Requirements
To run this project, ensure the following packages are installed:

Python 3.7+
PyTorch
torchvision
matplotlib
You can install all requirements via:

bash
Copy code
pip install torch torchvision matplotlib
Model Architecture
Generator
The Generator takes a latent vector as input and creates an image through a series of transposed convolutional layers. Batch normalization and ReLU activations are applied, with a final Tanh activation to generate pixel values between -1 and 1.

Discriminator
The Discriminator is a convolutional network that classifies images as real or fake. It uses spectral normalization and LeakyReLU activations to improve stability and learning efficiency.

Training the Model
The model is trained to minimize the losses:

Discriminator Loss: Maximizes the probability of correctly classifying real and fake images.
Generator Loss: Encourages the Generator to generate realistic images that fool the Discriminator.
The GAN is trained for several epochs on the MNIST dataset. Label smoothing and spectral normalization are applied to improve model performance.

Usage
Training the GAN
Run the following command to start training:

bash
Copy code
python gan_training_script.py
This script will:

Load and preprocess the MNIST dataset.
Train the Generator and Discriminator with the specified hyperparameters.
Display sample images generated after each epoch.
Parameters
You can modify the training parameters like num_epochs, lr, and batch_size in the script.

Results
Generated samples are displayed at the end of each epoch. For example:


The Generator's outputs gradually improve as the training progresses.

References
Goodfellow et al., GANs
DCGAN: Deep Convolutional Generative Adversarial Networks
