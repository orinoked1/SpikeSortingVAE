from dataHandle import SpikeDataLoader
from autoEncoder import Vae
import os
import torch
import numpy as np
import matplotlib.pyplot as plt



do_3_stage_train = True
if do_3_stage_train:

    batch_size = 2048
    shuffle = True
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.1"]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)

    cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
           "conv1_ch": 256, "conv2_ch": 16, "latent_dim": 36,
           "conv0_ker": 1, "conv1_ker": 3, "conv2_ker": 1,
           "ds_ratio_1": 2, "ds_ratio_2": 2,
           "cardinality": 32, "dropRate": 0.2, "n_epochs": 15,
           "learn_rate": 1e-3, "weight_decay": 1e-5,
           "shuffle_channels": False}
    cfg["ds_ratio_tot"] = cfg["ds_ratio_2"] * cfg["ds_ratio_1"]
    torch.manual_seed(0)
    np.random.seed(0)
    # training
    vae_model = Vae(cfg)
    unique_labels, target_means = vae_model.calc_means(data_loader)
    target_means = np.random.randn(target_means.shape[0], target_means.shape[1])
    target_means = target_means / np.linalg.norm(target_means, axis=1)[:, None]
    factors = [10,15,30]
    LRs = [3e-3, 3e-4]
    # train
    for fact in factors:
        for lr in LRs:
            vae_model = Vae(cfg)

            vae_model.cfg["n_epochs"] = 20
            cur_target_means = target_means * fact
            vae_model.optimizer = torch.optim.Adam(vae_model.parameters(),
                                                           lr=lr,
                                                           weight_decay=cfg["weight_decay"])


            vae_model.unique_labels = unique_labels
            vae_model.target_means = cur_target_means
            loss_array = vae_model.train_data_loader(data_loader)
            plt.plot(loss_array)
            plt.savefig(os.path.join(os.getcwd(),
                                     'vaeStage3_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}_F{}.png'.format(cfg["latent_dim"],
                                                                                           lr,
                                                                                           cfg["weight_decay"],
                                                                                           cfg["shuffle_channels"],
                                                                                           cfg["dropRate"],
                                                                                           fact)))
            plt.clf()
            vae_model.save_model(os.path.join(os.getcwd(),
                                              'vaeStage3_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}_F{}.pt'.format(
                                                  cfg["latent_dim"],
                                                  lr,
                                                  cfg["weight_decay"],
                                                  cfg["shuffle_channels"],
                                                  cfg["dropRate"],
                                                  fact)))
            del vae_model