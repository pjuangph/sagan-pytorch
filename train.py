from tqdm import tqdm
import numpy as np
import glob
import os
from PIL import Image

import argparse

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [       
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

def get_args_parser():
    parser = argparse.ArgumentParser(description='Self-Attention GAN trainer')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--workers', default=8, type=int, help='number of workers to load the data')
    parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
    parser.add_argument('--code', default=28, type=int, help='size of code to input generator')
    parser.add_argument('--lr_g', default=1e-4, type=float, help='learning rate of generator')
    parser.add_argument('--lr_d', default=4e-4, type=float, help='learning rate of discriminator')
    parser.add_argument('--n_d',default=1,type=int, help='number of discriminator update per 1 generator update')
    parser.add_argument('--model', default='dcgan', choices=['dcgan', 'resnet'], help='choice model class')
    parser.add_argument('--path', metavar='PATH', type=str, help='Path to image directory')
    parser.add_argument('--save_interval', default=100, type=int, help='Path to image directory')
    return parser



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(path, batch_size):
    dataset = datasets.MNIST(root=path,train=True,transform=transform,download=True)
    # dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    return loader


def train(args, n_class, generator, discriminator):
    dataset = datasets.MNIST(root=args.path,train=True,transform=transform,download=True)

    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)

    pbar = tqdm(range(args.iter), dynamic_ncols=True)

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    preset_code = torch.randn(n_class * 5, args.code).to(device)

    disc_loss_val = 0
    gen_loss_val = 0
    n = len(train_loader)
    for i in pbar:
        for batch_idx, (data,target) in enumerate(train_loader):
            discriminator.zero_grad()
            real_image, label =data, target
            b_size = real_image.size(0)
            real_image = real_image.to(device)
            fake_image = generator(
                torch.randn(b_size, args.code).to(device), label.to(device)
            )
            label = label.to(device)
            fake_predict = discriminator(fake_image, label)
            real_predict = discriminator(real_image, label)
            loss = F.relu(1 + fake_predict).mean()
            loss = loss + F.relu(1 - real_predict).mean()
            disc_loss_val = loss.detach().item()
            loss.backward()
            d_optimizer.step()

            if (i + 1) % args.n_d == 0:
                generator.zero_grad()
                requires_grad(generator, True)
                requires_grad(discriminator, False)
                input_class = torch.multinomial(
                    torch.ones(n_class), args.batch_size, replacement=True
                ).to(device)
                fake_image = generator(
                    torch.randn(args.batch_size, args.code).to(device), input_class
                )
                predict = discriminator(fake_image, input_class)
                loss = -predict.mean()
                gen_loss_val = loss.detach().item()
                loss.backward()
                g_optimizer.step()
                requires_grad(generator, False)
                requires_grad(discriminator, True)

            if (i + 1) % 100 == 0:                                  # Validation
                generator.train(False)
                input_class = torch.arange(n_class).long().repeat(5).to(device)
                fake_image = generator(preset_code, input_class)
                generator.train(True)
                utils.save_image(
                    fake_image.cpu().data,
                    f'sample/{str(i + 1).zfill(7)}.png',
                    nrow=n_class,
                    normalize=True,
                    range=(-1, 1),
                )

            if (i + 1) % args.save_interval == 0:
                no = str(i + 1).zfill(7)
                torch.save(generator.state_dict(), f'checkpoint/generator_{no}.pt')
                torch.save(discriminator.state_dict(), f'checkpoint/discriminator_{no}.pt')
                torch.save(g_optimizer.state_dict(), f'checkpoint/gen_optimizer_{no}.pt')
                torch.save(d_optimizer.state_dict(), f'checkpoint/dis_optimizer_{no}.pt')

            pbar.set_description(
                (f'Epoch: {i + 1}; G: {gen_loss_val:.5f};' f' D: {disc_loss_val:.5f} P: {100*batch_idx/n:1f}%')
            )


if __name__ == '__main__':
    parser =get_args_parser()
    args = parser.parse_args()
    print(args)

    n_class = 10 # MNIST uses 10 classes

    if args.model == 'dcgan':
        from model import Generator, Discriminator

    elif args.model == 'resnet':
        from model_resnet import Generator, Discriminator

    generator = Generator(image_size=args.code, n_class=n_class).to(device)
    discriminator = Discriminator(n_class).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0, 0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0, 0.9))
    train(args, n_class, generator, discriminator)
