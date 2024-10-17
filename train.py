import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from settings import args
import logging
from torch.utils.tensorboard import SummaryWriter
from dataloader import cell_dataloader
import numpy as np
from diffusion import DiffusionGene
from transformer import DiT
from cratetrans import WBDiT


def train_ddpm(args):
    device = args.device
    if args.model == 'DiT':
        model = DiT(depth=args.depth).to(device)
    else:
        model = WBDiT(depth=args.depth).to(device)

    train_loss_list = list()

    dataloader = cell_dataloader

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)

    if args.ckpt:
        model.load_state_dict(torch.load(f'ckpt/{args.model}/{args.run_name}/_epoch{args.ckpt_epoch}_ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'optimizer/{args.model}/{args.run_name}_AdamW.pt'))

    logger = SummaryWriter(os.path.join("runs", args.model, args.run_name))

    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss_list = list()

        for i, genes in enumerate(pbar):
            genes = genes.to(device)
            t = diffusion.sample_timesteps(genes.shape[0]).to(device)
            x_t, noise = diffusion.noise_genes(genes, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            epoch_loss_list.append(loss.item())

        avg_epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        train_loss_list.append(avg_epoch_loss)

        print("epoch: ", epoch, " avg_loss: ", avg_epoch_loss)

        if (epoch + 1) % args.save_every_epoch == 0:
            if args.ckpt == 'False':
                torch.save(model.state_dict(), os.path.join("ckpt", args.model, args.run_name, "_epoch{}_ckpt.pt".format(epoch + 1)))
            else:
                torch.save(model.state_dict(), os.path.join("ckpt", args.model, args.run_name, "_epoch{}_ckpt.pt".format(args.ckpt_epoch + epoch + 1)))
            torch.save(optimizer.state_dict(), f'optimizer/{args.model}/{args.run_name}_AdamW.pt')

    save_loss = np.array(train_loss_list)
    if args.ckpt == 'False':
        np.save(f'loss/{args.model}/{args.run_name}_0-{args.epochs}', save_loss)
    else:
        np.save(f'loss/{args.model}/{args.run_name}_{args.ckpt_epoch}-{args.ckpt_epoch + args.epochs}', save_loss)


if __name__ == '__main__':
    train_ddpm(args)
