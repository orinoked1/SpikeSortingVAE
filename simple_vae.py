import torch
from torch import nn
import numpy as np
from torch import optim
from utils import get_padding
from AutoEncoderBaseClass import AutoEncoderBaseClass


class SimpleVae(AutoEncoderBaseClass):
    def __init__(self, cfg):
        super(SimpleVae, self).__init__(cfg)

        self.enc_conv_1 = nn.Sequential(
            nn.Conv2d(1, cfg["enc_conv_1_ch"], (cfg["conv_ker"], cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]), get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["enc_conv_1_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.enc_conv_2 = nn.Sequential(
            nn.Conv2d(cfg["enc_conv_1_ch"], cfg["enc_conv_2_ch"], (cfg["conv_ker"], cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]), get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["enc_conv_2_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.enc_conv_4 = nn.Sequential(
            nn.Conv2d(cfg["enc_conv_2_ch"], cfg["enc_conv_4_ch"] * 2, (cfg["conv_ker"], cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]), get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["enc_conv_4_ch"] * 2),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.ds_2_2 = nn.MaxPool2d((cfg["ds_ratio"], cfg["ds_ratio"]))
        self.n_recording_chan_1 = cfg["n_channels"]
        self.n_recording_chan_2 = np.floor(
            (self.n_recording_chan_1 + 2 * get_padding(cfg["ds_ratio"]) - cfg["ds_ratio"]) / cfg["ds_ratio"]).astype(
            'int') + 1
        self.n_recording_chan_3 = np.floor(
            (self.n_recording_chan_2 + 2 * get_padding(cfg["ds_ratio"]) - cfg["ds_ratio"]) / cfg["ds_ratio"]).astype(
            'int') + 1
        self.n_recording_chan_4 = np.floor(
            (self.n_recording_chan_3 + 2 * get_padding(cfg["ds_ratio"]) - cfg["ds_ratio"]) / cfg["ds_ratio"]).astype(
            'int') + 1

        self.spk_length_1 = cfg["spk_length"]
        self.spk_length_2 = np.floor(
            (self.spk_length_1 + 2 * get_padding(cfg["ds_ratio"]) - cfg["ds_ratio"]) / cfg["ds_ratio"]).astype(
            'int') + 1
        self.spk_length_3 = np.floor(
            (self.spk_length_2 + 2 * get_padding(cfg["ds_ratio"]) - cfg["ds_ratio"]) / cfg["ds_ratio"]).astype(
            'int') + 1
        self.spk_length_4 = np.floor(
            (self.spk_length_3 + 2 * get_padding(cfg["ds_ratio"]) - cfg["ds_ratio"]) / cfg["ds_ratio"]).astype(
            'int') + 1
        self.dec_conv_1 = nn.Sequential(
            nn.Conv2d(cfg["dec_conv_1_ch"], 1, (cfg["conv_ker"], cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]), get_padding(cfg["conv_ker"])), bias=False),
        )
        self.dec_conv_2 = nn.Sequential(
            nn.Conv2d(cfg["dec_conv_2_ch"], cfg["dec_conv_1_ch"], (cfg["conv_ker"], cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]), get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["dec_conv_1_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.dec_conv_4 = nn.Sequential(
            nn.Conv2d(cfg["dec_conv_4_ch"], cfg["dec_conv_2_ch"], (cfg["conv_ker"], cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]), get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["dec_conv_2_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )
        # move model to GPU
        if torch.cuda.is_available():
            self.cuda()
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.cfg["learn_rate"],
                                    weight_decay=self.cfg["weight_decay"], amsgrad=True)

    def encoder(self, x):
        if x.ndim == 3:
            x = x[:, None, :, :, ]
        out = self.enc_conv_1(x)
        out = self.ds_2_2(out)
        out = self.enc_conv_2(out)
        out = self.ds_2_2(out)
        out = self.enc_conv_4(out)
        mu, log_var = torch.split(out, round(out.shape[1] / 2), dim=1)
        return mu, log_var

    def decoder(self, x):
        out = self.dec_conv_4(x)
        out = nn.functional.interpolate(out, size=(self.n_recording_chan_2, self.spk_length_2), mode='bicubic')
        out = self.dec_conv_2(out)
        out = nn.functional.interpolate(out, size=(self.n_recording_chan_1, self.spk_length_1), mode='bicubic')
        out = self.dec_conv_1(out)
        return out

# from dataHandle import SpikeDataLoader
# import os
# batch_size = 1024
# shuffle = True
# file_dirs = ["C:/DL_data"]
# file_clu_names = ["mF105_10.spk.1"]
# data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)
#
# simple_cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
#           "enc_conv_1_ch": 32, "enc_conv_2_ch": 64, "enc_conv_3_ch": 128, "enc_conv_4_ch": 8,
#           "dec_conv_1_ch": 32, "dec_conv_2_ch": 64, "dec_conv_3_ch": 128, "dec_conv_4_ch": 8,
#           "conv_ker": 3,
#           "ds_ratio": 2,
#           "cardinality": 32, "dropRate": 0.2, "n_epochs": 15,
#           "learn_rate": 1e-3, "weight_decay": 1e-5}
# n_epochs = 100
# vae_model = simple_vae(simple_cfg)
# # vae_model.train_data_loader(data_loader)
# # vae_model.save_model(os.path.join(os.getcwd(), 'simple_vae.pt'))
# vae_model = simple_vae.load_vae_model(r"simple_vae_stage_1_LD1_LR1E-03_WD1E-05_DR0.2.pt")
# vae_model.calc_means(data_loader)
#
