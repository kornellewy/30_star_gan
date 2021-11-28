import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from pathlib import Path

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter


class StarGan:
    def __init__(self, params: dict, dataset_path: str) -> None:
        self.writer = SummaryWriter()

        self.epoch = params["epoch"]
        self.n_epochs = params["n_epochs"]
        self.dataset_name = params["dataset_name"]
        self.batch_size = params["batch_size"]
        self.lr = params["lr"]
        self.b1 = params["b1"]
        self.b2 = params["b2"]
        self.decay_epoch = params["decay_epoch"]
        self.n_cpu = params["n_cpu"]
        self.img_height = params["img_height"]
        self.img_width = params["img_width"]
        self.channels = params["channels"]
        self.sample_interval = params["sample_interval"]
        self.checkpoint_interval = params["checkpoint_interval"]
        self.residual_blocks = params["residual_blocks"]
        self.selected_attrs = params["selected_attrs"]
        self.c_dim = len(self.selected_attrs)
        self.n_critic = params["n_critic"]
        self.lambda_cls = params["lambda_cls"]
        self.lambda_rec = params["lambda_rec"]
        self.lambda_gp = params["lambda_gp"]
        self.img_shape = (self.channels, self.img_height, self.img_width)

        self.train_images_save_dir_path = "images"
        self.train_models_save_dir_path = "saved_models"

        self.cuda = torch.cuda.is_available()
        self.criterion_cycle = torch.nn.L1Loss()
        self.generator, self.discriminator = self.load_models()
        self.generator_scaler = torch.cuda.amp.GradScaler()
        self.discriminator_scaler = torch.cuda.amp.GradScaler()

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )

        self.dataset_path = dataset_path
        self.train_transforms = [
            transforms.Resize(int(1.12 * self.img_height), Image.BICUBIC),
            transforms.RandomCrop(self.img_height),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.train_dataloader = DataLoader(
            CelebADataset(
                dataset_path,
                transforms_=self.train_transforms,
                mode="train",
                attributes=self.selected_attrs,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
            pin_memory=True,
        )
        self.val_transforms = [
            transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.val_dataloader = DataLoader(
            CelebADataset(
                dataset_path,
                transforms_=self.val_transforms,
                mode="val",
                attributes=self.selected_attrs,
            ),
            batch_size=10,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        self.tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.label_changes = [
            ((0, 1), (1, 0), (2, 0)),  # Set to black hair
            ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
            ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
            ((3, -1),),  # Flip gender
            ((4, -1),),  # Age flip
        ]

    def _create_dir_structure(self) -> None:
        Path(self.train_images_save_dir_path).mkdir(parents=True, exist_ok=True)
        Path(self.train_models_save_dir_path).mkdir(parents=True, exist_ok=True)

    def load_models(self):
        generator = GeneratorResNet(
            img_shape=self.img_shape, res_blocks=self.residual_blocks, c_dim=self.c_dim
        )
        discriminator = Discriminator(img_shape=self.img_shape, c_dim=self.c_dim)

        if self.cuda:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            self.criterion_cycle.cuda()
        if self.epoch != 0:
            # Load pretrained models
            generator.load_state_dict(
                torch.load(
                    str(
                        Path(self.train_models_save_dir_path).joinpath(
                            f"generator_{self.epoch}.pth"
                        )
                    )
                )
            )
            discriminator.load_state_dict(
                torch.load(
                    str(
                        Path(self.train_models_save_dir_path).joinpath(
                            f"discriminator_{self.epoch}.pth"
                        )
                    )
                )
            )
        else:
            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)
        return generator, discriminator

    @staticmethod
    def criterion_cls(logit, target):
        return F.binary_cross_entropy_with_logits(
            logit, target, size_average=False
        ) / logit.size(0)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates, _ = self.discriminator(interpolates)
        fake = Variable(self.tensor(np.ones(d_interpolates.shape)), requires_grad=False)
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

    def sample_images(self, batches_done):
        """Saves a generated sample of domain translations"""
        val_imgs, val_labels = next(iter(self.val_dataloader))
        val_imgs = Variable(val_imgs.type(self.tensor))
        val_labels = Variable(val_labels.type(self.tensor))
        img_samples = None
        for i in range(10):
            img, label = val_imgs[i], val_labels[i]
            # Repeat for number of label changes
            imgs = img.repeat(self.c_dim, 1, 1, 1)
            labels = label.repeat(self.c_dim, 1)
            # Make changes to labels
            for sample_i, changes in enumerate(self.label_changes):
                for col, val in changes:
                    labels[sample_i, col] = (
                        1 - labels[sample_i, col] if val == -1 else val
                    )

            # Generate translations
            gen_imgs = self.generator(imgs, labels)
            # Concatenate images by width
            gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
            img_sample = torch.cat((img.data, gen_imgs), -1)
            # Add as row to generated samples
            img_samples = (
                img_sample
                if img_samples is None
                else torch.cat((img_samples, img_sample), -2)
            )

        save_image(
            img_samples.view(1, *img_samples.shape),
            f"{self.train_images_save_dir_path}/{batches_done}.png",
            normalize=True,
        )

    def train(self):
        start_time = time.time()
        for epoch in range(self.epoch, self.n_epochs):
            for i, (imgs, labels) in enumerate(self.train_dataloader):
                # Model inputs
                imgs = Variable(imgs.type(self.tensor))
                labels = Variable(labels.type(self.tensor))
                # Sample labels as generator inputs
                # torch.Size([16, 5]), same 1
                sampled_c = Variable(
                    self.tensor(np.random.randint(0, 2, (imgs.size(0), self.c_dim)))
                )
                # Generate fake batch of images
                with torch.cuda.amp.autocast():
                    fake_imgs = self.generator(imgs, sampled_c)
                    #  Train Discriminator
                    self.optimizer_D.zero_grad()
                    # Real images
                    real_validity, pred_cls = self.discriminator(imgs)
                    # Fake images
                    fake_validity, _ = self.discriminator(fake_imgs.detach())
                    # Gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(
                        imgs.data, fake_imgs.data
                    )
                    # Adversarial loss
                    loss_D_adv = (
                        -torch.mean(real_validity)
                        + torch.mean(fake_validity)
                        + self.lambda_gp * gradient_penalty
                    )
                    # Classification loss
                    loss_D_cls = self.criterion_cls(pred_cls, labels)
                    # Total loss
                    loss_D = loss_D_adv + self.lambda_cls * loss_D_cls

                self.generator_scaler.scale(loss_D).backward()
                self.generator_scaler.step(self.optimizer_D)
                self.generator_scaler.update()

                # loss_D.backward()
                # self.optimizer_D.step()

                iteration = epoch * len(self.train_dataloader) + i
                self.writer.add_scalar(
                    "gradient_penalty/train_step", gradient_penalty, iteration
                )
                self.writer.add_scalar("loss_D_adv/train_step", loss_D_adv, iteration)
                self.writer.add_scalar("loss_D_cls/train_step", loss_D_cls, iteration)
                self.writer.add_scalar("loss_D/train_step", loss_D, iteration)

                self.optimizer_G.zero_grad()

                # Every n_critic times update generator
                if i % self.n_critic == 0:
                    #  Train Generator
                    # Translate and reconstruct image
                    with torch.cuda.amp.autocast():
                        gen_imgs = self.generator(imgs, sampled_c)
                        recov_imgs = self.generator(gen_imgs, labels)
                        # Discriminator evaluates translated image
                        fake_validity, pred_cls = self.discriminator(gen_imgs)
                        # Adversarial loss
                        loss_G_adv = -torch.mean(fake_validity)
                        # Classification loss
                        loss_G_cls = self.criterion_cls(pred_cls, sampled_c)
                        # Reconstruction loss
                        loss_G_rec = self.criterion_cycle(recov_imgs, imgs)
                        # Total loss
                        loss_G = (
                            loss_G_adv
                            + self.lambda_cls * loss_G_cls
                            + self.lambda_rec * loss_G_rec
                        )

                    self.discriminator_scaler.scale(loss_G).backward()
                    self.discriminator_scaler.step(self.optimizer_G)
                    self.discriminator_scaler.update()

                    # loss_G.backward()
                    # self.optimizer_G.step()

                    self.writer.add_scalar(
                        "loss_G_rec/train_step", loss_G_rec, iteration
                    )
                    self.writer.add_scalar(
                        "loss_G_cls/train_step", loss_G_cls, iteration
                    )
                    self.writer.add_scalar(
                        "loss_G_adv/train_step", loss_G_adv, iteration
                    )
                    self.writer.add_scalar("loss_G/train_step", loss_G, iteration)
                    #  Log Progress
                    # Determine approximate time left
                    batches_done = epoch * len(self.train_dataloader) + i
                    batches_left = (
                        self.n_epochs * len(self.train_dataloader) - batches_done
                    )
                    time_left = datetime.timedelta(
                        seconds=batches_left
                        * (time.time() - start_time)
                        / (batches_done + 1)
                    )
                    # # Print log
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f] ETA: %s"
                        % (
                            epoch,
                            self.n_epochs,
                            i,
                            len(self.train_dataloader),
                            loss_D_adv.item(),
                            loss_D_cls.item(),
                            loss_G.item(),
                            loss_G_adv.item(),
                            loss_G_cls.item(),
                            loss_G_rec.item(),
                            time_left,
                        )
                    )

                    # If at sample interval sample and save image
                    if batches_done % self.sample_interval == 0:
                        self.sample_images(batches_done)

            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(
                    self.generator.state_dict(), "saved_models/generator_%d.pth" % epoch
                )
                torch.save(
                    self.discriminator.state_dict(),
                    "saved_models/discriminator_%d.pth" % epoch,
                )


if __name__ == "__main__":
    dataset_path = "J:/kjn_YT/29_cycle_gan_black_white/CelebA/Img/img_align_celeba/img_align_celeba"
    params = {
        "epoch": 14,
        "n_epochs": 1000,
        "dataset_name": "img_align_celeba",
        "batch_size": 32,
        "lr": 0.0002,
        "b1": 0.5,
        "b2": 0.999,
        "decay_epoch": 100,
        "n_cpu": 10,
        "img_height": 128,
        "img_width": 128,
        "channels": 3,
        "sample_interval": 400,
        "checkpoint_interval": 1,
        "residual_blocks": 6,
        "selected_attrs": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
        "n_critic": 5,
        "lambda_cls": 1,
        "lambda_rec": 10,
        "lambda_gp": 10,
    }
    kjn = StarGan(params=params, dataset_path=dataset_path)
    kjn.train()
