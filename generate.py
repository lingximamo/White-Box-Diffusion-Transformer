from diffusion import DiffusionGene
from settings import args
import numpy as np
import torch
from transformer import DiT
from cratetrans import WBDiT


def sample_batches(model, amount=1024, savepath=None, method='ddpm', sub_time_seq=None, nonnagative=True, eta=1.):
    device = args.device
    num = args.batch_size
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)
    batches = None
    rest = amount % num
    for i in range(amount // num):
        print(f'generating batch {i + 1} ...')
        if method == 'ddpm':
            batch = diffusion.sample(model, n=num, nonnagative=nonnagative).to('cpu')
        else:
            batch = diffusion.sample_ddim(model, n=num, eta=eta, sub_time_seq=sub_time_seq, nonnagative=nonnagative).to('cpu')
        if batches is None:
            batches = batch
        else:
            batches = torch.cat((batches, batch), dim=0)
        print(batches.shape)
        if savepath:
            np.save(savepath, batches.numpy())

    if rest != 0:
        print(f'generating last {rest} samples...')
        if method == 'ddpm':
            batch = diffusion.sample(model, n=rest, nonnagative=nonnagative).to('cpu')
        else:
            batch = diffusion.sample_ddim(model, n=rest, eta=eta, sub_time_seq=sub_time_seq, nonnagative=nonnagative).to('cpu')
        if batches is None:
            batches = batch
        else:
            batches = torch.cat((batches, batch), dim=0)
        print(batches.shape)
        if savepath:
            np.save(savepath, batches.numpy())


if __name__ == '__main__':
    device = args.device

    ckpt_path = 'ckpt/WBDiT/Gene_fibroblast/_epoch3_ckpt.pt'     # path of checkpoint
    number = 10     # number of generated samples

    method = 'ddim'     # use ddim or ddpm
    # method = 'ddpm'

    acc_rate = 10  # ddim accelerate rate
    save = 'results/WBDiT/Gene_fibroblast_samples'  # savepath of result
    nonnegative = True     # if dataset is non-negative, set True

    model = WBDiT(depth=args.depth).to(device)     # choose model: DiT or WBDiT
    # model = DiT(depth=args.depth).to(device)

    model.load_state_dict(torch.load(ckpt_path))

    # generate subsequence(ddim)
    sub_time_seq = [i for i in range(0, 1001, acc_rate)]
    sub_time_seq.insert(1, 1)

    sample_batches(model, amount=number, savepath=save, method=method, sub_time_seq=sub_time_seq, nonnagative=nonnegative, eta=1)

