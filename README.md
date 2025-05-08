# White-Box Diffusion Transformer for single-cell RNA-seq generation

## Dataset Preparation
-   Place your gene dataset (`.npy`  format) in the  `datasets/`  directory.
-   Supported datasets:  `fibroblast_datas.npy`,  `malignant_datas.npy`, or  `immune_datas.npy`.
-   Update  `settings.py`  to specify the dataset path and name.
-   The default dimension of cell gene is 1D arrays of length 2000 in `.npy` format. You can change the dimension of the cells by modifying the `gene_size` parameter in `settings.py`.

## Training

1.  Configure training settings in  `settings.py`: 
    -   Choose model:  `args.model = 'WBDiT'`  or  `'DiT'`.
    -   Set parameters such as `batch_size`,  `depth`  (number of transformer blocks), and  `epochs`.

2.  Run training:
    ```
    python train.py
    ```
3.  Checkpoints and loss logs are saved in  `ckpt/`  and  `loss/`  directories.

4.  Example (parameters of `settings.py`):
    ```
    args.run_name = 'Gene_fibroblast' # dataset name
    args.dataset_path = 'datasets/fibroblast_datas.npy' # path of dataset
    args.model = 'WBDiT'  # choose model: 'WBDiT' or 'DiT'
    args.depth = 3     # number of transformer blocks
    args.epochs = 3     # training epochs
    args.batch_size = 16     # batch size
    args.gene_size = 2000     # size of gene
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lr = 1e-4     # learning rate
    args.ckpt = False     # load checkpoint or not
    args.ckpt_epoch = 0     # if args.ckpt == True, choose which checkpoint to load
    args.save_every_epoch = 3     # save every n epoch
    ```
## Generation
1.  Update  `generate.py`:
    -   Specify checkpoint path (e.g.,  `ckpt/WBDiT/Gene_fibroblast/_epoch3_ckpt.pt`).
    -   Choose sampling method (`method = 'ddim'`  or  `'ddpm'`).
    -   Modify the `number` parameter (number of generated samples)
        
2.  Generate samples:
    ```
    python generate.py
    ```
3.  Generated samples are saved to  `results/`.

4.  Example (parameters of `generate.py`):
    ```
    ckpt_path = 'ckpt/WBDiT/Gene_fibroblast/_epoch3_ckpt.pt'     # path of checkpoint
    number = 10     # number of generated samples
    method = 'ddim'     # use ddim or ddpm
    acc_rate = 10  # ddim accelerate rate
    save = 'results/WBDiT/Gene_fibroblast_samples'  # savepath of result
    nonnegative = True     # if dataset is non-negative, set True
    model = WBDiT(depth=args.depth).to(device)     # choose model: DiT or WBDiT
    ```
    


