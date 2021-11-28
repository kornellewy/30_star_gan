"""
source:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/stargan/stargan.py
https://arxiv.org/pdf/1711.09020.pdf
https://www.youtube.com/watch?v=Lbo7AWPeA54
"""

import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult

from models import *
from datasets import *
from optims.Adam import Adam_GCC2


class StarGanTrainLoop(pl.loops.OptimizerLoop):
    def __init__(self) -> None:
        super().__init__()

    def _optimizer_step(
        self,
        optimizer: Optimizer,
        opt_idx: int,
        batch_idx: int,
        train_step_and_backward_closure: Callable[[], Optional[Tensor]],
    ) -> None:
        """Performs the optimizer step and some sanity checking.
        Args:
            optimizer: the optimizer to perform the step with
            opt_idx: the index of the current :param:`optimizer`
            batch_idx: the index of the current batch
            train_step_and_backward_closure: the closure function performing the train step and computing the
                gradients. By default called by the optimizer (if possible)
        """
        lightning_module = self.trainer.lightning_module

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        # wraps into LightningOptimizer only for running step
        # optimizer = LightningOptimizer._to_lightning_optimizer(
        #     optimizer, self.trainer, opt_idx
        # )

        self.optim_progress.optimizer.step.increment_ready()

        # model hook
        # lightning_module.optimizer_step(
        #     self.trainer.current_epoch,
        #     batch_idx,
        #     optimizer,
        #     opt_idx,
        #     train_step_and_backward_closure,
        #     on_tpu=(
        #         self.trainer._device_type == _AcceleratorType.TPU and _TPU_AVAILABLE
        #     ),
        #     using_native_amp=(
        #         self.trainer.amp_backend is not None
        #         and self.trainer.amp_backend == AMPType.NATIVE
        #     ),
        #     using_lbfgs=is_lbfgs,
        # )

        self.optim_progress.optimizer.step.increment_completed()
        optimizer.step(closure=optimizer_closure)


class StarGan(pl.LightningModule):
    def __init__(
        self, hparams, residual_blocks, img_shape, c_dim, n_critic, val_loader
    ):
        super().__init__()

        self._hparams = hparams
        self.img_shape = img_shape
        self.c_dim = c_dim
        self._model_generator = GeneratorResNet(
            img_shape=img_shape, res_blocks=residual_blocks, c_dim=c_dim
        )
        self._model_discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim)

        self._model_generator = self._model_generator.to(self.device)
        self._model_discriminator = self._model_discriminator.to(self.device)

        self._model_generator.apply(weights_init_normal)
        self._model_discriminator.apply(weights_init_normal)

        self.criterion_cycle = torch.nn.L1Loss()
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.lambda_gp = 10

        self.discriminator_update_count = 0
        self.n_critic = n_critic

        self.val_loader = val_loader
        self.Tensor = torch.cuda.FloatTensor
        # self.train_loop = StarGanTrainLoop(
        #     g_model, d_model, g_optimizers, d_optimizers, dataloader
        # )

    def criterion_cls(self, logit, target):
        return F.binary_cross_entropy_with_logits(
            logit, target, size_average=False
        ) / logit.size(0)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates, _ = self._model_discriminator(interpolates)
        fake = Variable(self.Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
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

        optimizer_generator = torch.optim.Adam(
            self._model_generator.parameters(), lr=lr, betas=(b1, b2)
        )
        optimizer_discriminator = torch.optim.Adam(
            self._model_discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        return (
            {"optimizer": optimizer_generator, "frequency": self.n_critic},
            {"optimizer": optimizer_discriminator, "frequency": 1},
        )

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_nb, optimizer_idx):

        imgs = batch[0]
        labels = batch[1]
        # Sample labels as generator inputs
        sampled_c = Variable(
            self.Tensor(np.random.randint(0, 2, (imgs.size(0), c_dim)))
        )
        # Generate fake batch of images
        with torch.no_grad():
            fake_imgs = self._model_generator(imgs, sampled_c)
        if optimizer_idx == 0:
            self._model_discriminator.zero_grad()
            # Real images
            self.optimizers()[1].zero_grad()
            self.optimizers()[0].zero_grad()
            fake_imgs = fake_imgs.detach()
            real_validity, pred_cls = self._model_discriminator(imgs.detach())
            # Fake images
            fake_validity, _ = self._model_discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)
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
            self.log("gradient_penalty", gradient_penalty)
            self.log("loss_D_adv", loss_D_adv)
            self.log("loss_D_cls", loss_D_cls)
            self.log("loss_D", loss_D)
            loss_D.backward()
            self.optimizers()[1].step()
            return None
        elif optimizer_idx == 1:
            self.optimizers()[1].zero_grad()
            self.optimizers()[0].zero_grad()
            gen_imgs = self._model_generator(imgs, sampled_c)
            recov_imgs = self._model_generator(gen_imgs, labels)
            # Discriminator evaluates translated image
            fake_validity, pred_cls = self._model_discriminator(gen_imgs)
            # Adversarial loss
            loss_G_adv = -torch.mean(fake_validity)
            # Classification loss
            loss_G_cls = self.criterion_cls(pred_cls, sampled_c)
            # Reconstruction loss
            loss_G_rec = self.criterion_cycle(recov_imgs, imgs)
            # Total loss
            loss_G = (
                loss_G_adv + self.lambda_cls * loss_G_cls + self.lambda_rec * loss_G_rec
            )
            self.log("loss_G_adv", loss_G_adv)
            self.log("loss_G_rec", loss_G_rec)
            self.log("loss_G_cls", loss_G_cls)
            self.log("loss_G", loss_G)
            print("loss_G", loss_G)
            loss_G.backward()
            self.optimizers()[0].step()
            return None

    # def validation_step(self, batch, batch_nb):
    # imgs = batch[0]
    # labels = batch[1]
    # # Sample labels as generator inputs
    # sampled_c = Variable(
    #     torch.tensor(
    #         np.random.randint(0, 2, (imgs.size(0), self.c_dim)),
    #         device=self.device,
    #     )
    # )
    # # Generate fake batch of images
    # fake_imgs = self._model_generator(imgs, sampled_c)
    # self.fake_imgs = fake_imgs
    # recov_imgs = self._model_generator(fake_imgs, labels)
    # # Discriminator evaluates translated image
    # fake_validity, pred_cls = self._model_discriminator(recov_imgs)
    # # Adversarial loss
    # loss_G_adv = -torch.mean(fake_validity)
    # # Classification loss
    # loss_G_cls = criterion_cls(pred_cls, sampled_c)
    # # Reconstruction loss
    # loss_G_rec = self.criterion_cycle(recov_imgs, imgs)
    # # Total loss
    # generator_loss = (
    #     loss_G_adv + self.lambda_cls * loss_G_cls + self.lambda_rec * loss_G_rec
    # )
    # self.log("valid_generator_loss", generator_loss)
    # all_images_sample = torch.cat((imgs, fake_imgs), -2)
    # self.sample_images(self.discriminator_update_count)
    # return generator_loss

    def sample_images(self, batches_done):
        """Saves a generated sample of domain translations"""
        label_changes = [
            ((0, 1), (1, 0), (2, 0)),  # Set to black hair
            ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
            ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
            ((3, -1),),  # Flip gender
            ((4, -1),),  # Age flip
        ]
        val_imgs, val_labels = next(iter(self.val_loader))
        val_imgs = Variable(val_imgs.type(torch.float32)).to(self.device)
        val_labels = Variable(val_labels.type(torch.float32)).to(self.device)
        img_samples = None
        for i in range(10):
            img, label = val_imgs[i], val_labels[i]
            # Repeat for number of label changes
            imgs = img.repeat(c_dim, 1, 1, 1)
            labels = label.repeat(c_dim, 1)
            # Make changes to labels
            for sample_i, changes in enumerate(label_changes):
                for col, val in changes:
                    labels[sample_i, col] = (
                        1 - labels[sample_i, col] if val == -1 else val
                    )

            # Generate translations
            gen_imgs = self._model_generator(imgs, labels)
            # Concatenate images by width
            gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
            img_sample = torch.cat((img.data, gen_imgs), -1)
            # Add as row to generated samples
            img_samples = (
                img_sample
                if img_samples is None
                else torch.cat((img_samples, img_sample), -2)
            )
        # save_image(
        #     img_samples.view(1, *img_samples.shape),
        #     "images/%s.png" % batches_done,
        #     normalize=True,
        #     value_range=(-1, 1),
        # )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # dataset_path = "J:/kjn_YT/29_cycle_gan_black_white/CelebA/Img/img_align_celeba/img_align_celeba"
    dataset_path = "CelebA_10000/Img/img_align_celeba/img_align_celeba"
    log_dir_path = ""

    selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    residual_blocks = 6
    hparams = {"batch_size": 40, "lr": 0.0002, "b1": 0.5, "b2": 0.999}
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
        num_workers=1,
        pin_memory=True,
    )

    val_transforms = [
        transforms.Resize((img_shape[1], img_shape[2]), Image.BICUBIC),
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
        num_workers=10,
        pin_memory=True,
    )

    pytorch_light_module = StarGan(
        hparams=hparams,
        residual_blocks=residual_blocks,
        img_shape=img_shape,
        c_dim=c_dim,
        n_critic=n_critic,
        val_loader=val_loader,
    )
    trainer = pl.Trainer(
        gpus=1,
        # precision=16,
        check_val_every_n_epoch=10,
        max_epochs=10,
        default_root_dir=log_dir_path,
    )
    # trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=StarGanTrainLoop())
    trainer.fit(
        pytorch_light_module,
        train_loader,
        val_loader,
    )
