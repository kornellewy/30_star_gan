"""
source:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/stargan/stargan.py
https://arxiv.org/pdf/1711.09020.pdf
"""

import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from models import *
from datasets import *
from optims.Adam import Adam_GCC, Adam_GCC2


def criterion_cls(logit, target):
    target = target.half()
    return F.binary_cross_entropy_with_logits(
        logit, target, size_average=False
    ) / logit.size(0)


class StarGan(pl.LightningModule):
    def __init__(
        self,
        hparams,
        residual_blocks,
        img_shape,
        c_dim,
        n_critic,
    ):
        super().__init__()
        self._hparams = hparams
        self.img_shape = img_shape
        self.c_dim = c_dim
        self._model_generator = GeneratorResNet(
            img_shape=img_shape, res_blocks=residual_blocks, c_dim=c_dim
        )
        self._model_discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim)

        # self._model_generator.apply(weights_init_normal)
        # self._model_discriminator.apply(weights_init_normal)

        self._model_generator = self._model_generator.to(self.device)
        self._model_discriminator = self._model_discriminator.to(self.device)

        self.criterion_cycle = torch.nn.L1Loss()
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.lambda_gp = 10

        self.label_changes = [
            ((0, 1), (1, 0), (2, 0)),  # Set to black hair
            ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
            ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
            ((3, -1),),  # Flip gender
            ((4, -1),),  # Age flip
        ]

        self.discriminator_update_count = 0
        self.n_critic = n_critic

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(
            np.random.random((real_samples.size(0), 1, 1, 1)), device=self.device
        )
        # Get random interpolation between real and fake samples
        interpolates = (
            (alpha * real_samples + ((1 - alpha) * fake_samples))
            .requires_grad_(True)
            .half()
        )
        d_interpolates, _ = self._model_discriminator(interpolates)
        fake = Variable(
            torch.tensor(np.ones(d_interpolates.shape), device=self.device),
            requires_grad=False,
        )
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def configure_optimizers(self):
        lr = self._hparams["lr"]
        b1 = self._hparams["b1"]
        b2 = self._hparams["b2"]

        optimizer_generator = Adam_GCC2(
            self._model_generator.parameters(), lr=lr, betas=(b1, b2)
        )
        optimizer_discriminator = Adam_GCC2(
            self._model_discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        return [optimizer_generator, optimizer_discriminator], []

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_nb, optimizer_idx):
        self.last_batch = batch
        imgs = batch[0]
        labels = batch[1]
        if optimizer_idx == 0:
            # Sample labels as generator inputs
            sampled_c = Variable(
                torch.tensor(
                    np.random.randint(0, 2, (imgs.size(0), self.c_dim)),
                    device=self.device,
                )
            )
            # Generate fake batch of images
            fake_imgs = self._model_generator(imgs, sampled_c)
            self.fake_imgs = fake_imgs
            # Every n_critic times update generator
            if self.discriminator_update_count % self.n_critic == 0:
                recov_imgs = self._model_generator(fake_imgs, labels)
                # Discriminator evaluates translated image
                fake_validity, pred_cls = self._model_discriminator(recov_imgs)
                # Adversarial loss
                loss_G_adv = -torch.mean(fake_validity)
                # Classification loss
                loss_G_cls = criterion_cls(pred_cls, sampled_c)
                # Reconstruction loss
                loss_G_rec = self.criterion_cycle(recov_imgs, imgs)
                # Total loss
                generator_loss = (
                    loss_G_adv
                    + self.lambda_cls * loss_G_cls
                    + self.lambda_rec * loss_G_rec
                )
                self.log("generator_loss", generator_loss)
                if self.discriminator_update_count % 1000 == 0:
                    all_images_sample = torch.cat((imgs, fake_imgs), -2)
                    save_image(
                        all_images_sample,
                        "images/all_%s.png" % batch_nb,
                        normalize=True,
                    )
                return generator_loss

        elif optimizer_idx == 1:
            real_validity, pred_cls = self._model_discriminator(imgs)
            fake_validity, _ = self._model_discriminator(self.fake_imgs.detach())
            gradient_penalty = self.compute_gradient_penalty(
                imgs.data, self.fake_imgs.data
            )
            # Adversarial loss
            loss_D_adv = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + self.lambda_gp * gradient_penalty
            )
            # Classification loss
            loss_D_cls = criterion_cls(pred_cls, labels)
            # Total loss
            discriminator_loss = loss_D_adv + self.lambda_cls * loss_D_cls
            self.log("discriminator_loss", discriminator_loss)
            self.discriminator_update_count += 1
            return discriminator_loss


if __name__ == "__main__":
    torch.cuda.empty_cache()
    dataset_path = "J:/kjn_YT/29_cycle_gan_black_white/CelebA/Img/img_align_celeba/img_align_celeba"
    log_dir_path = ""

    selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    residual_blocks = 6
    hparams = {"batch_size": 60, "lr": 0.0002, "b1": 0.5, "b2": 0.999}
    img_shape = (3, 128, 128)
    c_dim = len(selected_attrs)
    n_critic = 5

    train_transforms = [
        transforms.Resize(int(1.12 * img_shape[1]), Image.BICUBIC),
        transforms.RandomCrop(img_shape[1]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_loader = DataLoader(
        CelebADataset(
            dataset_path,
            transforms_=train_transforms,
            mode="train",
            attributes=selected_attrs,
        ),
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_transforms = [
        transforms.Resize((img_shape[1], img_shape[0]), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    val_loader = DataLoader(
        CelebADataset(
            dataset_path,
            transforms_=val_transforms,
            mode="val",
            attributes=selected_attrs,
        ),
        batch_size=10,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    pytorch_light_module = StarGan(
        hparams=hparams,
        residual_blocks=residual_blocks,
        img_shape=img_shape,
        c_dim=c_dim,
        n_critic=n_critic,
    )
    trainer = pl.Trainer(
        gpus=-1,
        precision=16,
        benchmark=True,
        check_val_every_n_epoch=1,
        max_epochs=200,
        default_root_dir=log_dir_path,
        resume_from_checkpoint="lightning_logs/version_0/checkpoints/epoch=5-step=20063.ckpt"
    )
    # pytorch_light_module = load_from_checkpoint.load_from_checkpoint(
    #     "lightning_logs/version_0/checkpoints/epoch=5-step=20063.ckpt",
    # )
    trainer.fit(
        pytorch_light_module,
        train_loader,
        val_loader,
    )
