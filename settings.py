import argparse
import torch

parser = argparse.ArgumentParser()
args = parser.parse_args()

# choose dataset
args.run_name = 'Gene_fibroblast'
# args.run_name = 'Gene_malignant'
# args.run_name = 'Gene_immune'


# path of dataset
args.dataset_path = 'datasets/fibroblast_datas.npy'
# args.dataset_path = 'datasets/malignant_datas.npy'
# args.dataset_path = 'datasets/immune_datas.npy'


# Choose model
args.model = 'DiT'
# args.model = 'WBDiT'
args.depth = 3     # number of DiTBlock


args.epochs = 2400     # training epochs
args.batch_size = 16     # batch size
args.gene_size = 2000     # size of gene
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.lr = 1e-4     # learning rate
args.ckpt = False     # load checkpoint or not
args.ckpt_epoch = 0     # if args.ckpt == True, choose which checkpoint to load
args.save_every_epoch = 100     # save every n epoch


