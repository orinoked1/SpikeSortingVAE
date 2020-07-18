import torch
from torch import nn
import torch.nn.functional as F
from utils import make_layers, get_padding, flatten, un_flatten
from reslib import ResNeXtBottleNeck, BasicResBlock
from torch import optim
from AutoEncoderBaseClass import AutoEncoderBaseClass

class Vae(AutoEncoderBaseClass):
    def __init__(self, cfg):
        super(Vae, self).__init__(cfg)

        # encoder
        self.res0 = make_layers(cfg["n_channels"], cfg["conv1_ch"], cfg["conv0_ker"], n_layers=1, cardinality=1,
                                dropRate=0)
        self.resx = ResNeXtBottleNeck(cfg["conv1_ch"], cfg["conv1_ker"], cardinality=cfg["cardinality"],
                                      dropRate=cfg["dropRate"])
        self.res2 = nn.Sequential(
            nn.Conv1d(cfg["conv1_ch"], cfg["conv2_ch"], cfg["conv2_ker"], groups=1,
                      padding=get_padding(cfg["conv2_ker"]), bias=False),
            nn.BatchNorm1d(cfg["conv2_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.enc_mu = nn.Linear(
            in_features=int(cfg["conv2_ch"] * cfg["spk_length"] / cfg["ds_ratio_tot"]),
            out_features=cfg["latent_dim"])
        self.enc_log_var = nn.Linear(
            in_features=int(cfg["conv2_ch"] * cfg["spk_length"] / cfg["ds_ratio_tot"]),
            out_features=cfg["latent_dim"])
        # decoder
        self.dec_linear = nn.Linear(
            in_features=cfg["latent_dim"],
            out_features=int(cfg["conv2_ch"] * cfg["spk_length"] / cfg["ds_ratio_tot"]))
        self.deres2 = make_layers(cfg["conv2_ch"], cfg["conv1_ch"], cfg["conv2_ker"], n_layers=1, decode=True,
                                  dropRate=cfg["dropRate"])
        self.deres1 = BasicResBlock(cfg["conv1_ch"], cfg["conv1_ker"], n_layers=2, decode=True,
                                    dropRate=cfg["dropRate"])
        self.deres0 = nn.ConvTranspose1d(cfg["conv1_ch"], cfg["n_channels"], cfg["conv0_ker"],
                                         padding=get_padding(cfg["conv0_ker"]))
        # down sampling layers
        self.ds1 = nn.MaxPool1d(cfg["ds_ratio_1"])
        self.ds2 = nn.MaxPool1d(cfg["ds_ratio_2"])
        # move model to GPU
        if torch.cuda.is_available():
            self.cuda()
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.cfg["learn_rate"],
                                    weight_decay=self.cfg["weight_decay"], amsgrad=True)
        self.unique_labels = []
        self.target_means = []

    def encoder(self, x):
        out = self.res0(x)
        out = self.resx(out)
        out = self.ds1(out)
        out = self.resx(out)
        out = self.ds2(out)
        out = self.res2(out)
        out = flatten(out)
        mu = self.enc_mu(out)
        log_var = self.enc_log_var(out)
        return mu, log_var

    def decoder(self, x):
        out = self.dec_linear(x)
        out = un_flatten(out, (-1, self.cfg["conv2_ch"], int(self.cfg["spk_length"] / self.cfg["ds_ratio_tot"])))
        out = self.deres2(out)
        out = F.interpolate(out, scale_factor=self.cfg["ds_ratio_2"])
        out = self.deres1(out)
        out = F.interpolate(out, scale_factor=self.cfg["ds_ratio_1"])
        out = self.deres1(out)
        out = self.deres0(out)
        return out
