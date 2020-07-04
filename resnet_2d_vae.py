import torch
from torch import nn
import numpy as np
import time
import torch.nn.functional as F
from torch import optim
from dataHandle import show_two_spikes
from utils import  get_padding
from reslib import ResNeXtBottleNeck_2d, BasicResBlock_2d


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def flatten(x):
    return x.view(x.size()[0], -1)


def un_flatten(x, shape):
    return x.view(shape)


class ReParameterize(nn.Module):
    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # generate a iid standard normal same shape as s
        return mu + eps * std


# def calc_vae_loss(x, x_recon, enc_mu, enc_log_var):
#     Loss = nn.MSELoss(reduction='sum')
#     reconstruction_loss = Loss(x_recon, x)
#     kld_loss = -0.5 * torch.sum(1 + enc_log_var - enc_mu ** 2 - torch.exp(enc_log_var))
#     return kld_loss + reconstruction_loss
def calc_vae_loss(x, x_recon, enc_mu, enc_log_var, y, unique_labels, target_means):
    x_recon = x_recon.squeeze()
    Loss = nn.MSELoss(reduction='sum')
    reconstruction_loss = Loss(x_recon, x)
    # Loss = nn.BCELoss(reduction='sum')
    # reconstruction_loss = Loss(x_recon, x)
    # reconstruction_loss = torch.sum(-x * x_recon.log())
    kld_loss = -0.5 * torch.sum(1 + enc_log_var - enc_mu ** 2 - torch.exp(enc_log_var))
    # print('KL loss =',kld_loss )
    # print('reconstruction_loss =',reconstruction_loss )
    return kld_loss + reconstruction_loss


def calc_ori_loss2(x, x_recon, enc_mu, enc_log_var, y, unique_labels, target_means):
    if (isinstance(unique_labels, list)) or (isinstance(unique_labels, list)):
        return calc_vae_loss(x, x_recon, enc_mu, enc_log_var, y, unique_labels, target_means)
    x_recon = x_recon.squeeze()
    reconstruction_loss = F.mse_loss(x, x_recon, reduction='sum')
    kld_loss = 0
    classes = unique_labels
    for i_class in range(len(classes)):
        class_mus = enc_mu[(y == classes[i_class]).flatten(), :]
        class_log_vars = enc_log_var[(y == classes[i_class]).flatten(), :]

        class_mu = torch.tensor(target_means[i_class]).cuda()
        class_var = torch.ones(class_mu.shape).cuda()
        kld_loss += .5 * (torch.sum(torch.log(class_var) -
                                    class_log_vars + (torch.exp(class_log_vars) / class_var) +
                                    (class_mus - class_mu) ** 2 / class_var -
                                    1))

    return kld_loss + reconstruction_loss


class resnet_2d_vae(nn.Module):
    def __init__(self, cfg):
        super(resnet_2d_vae, self).__init__()

        self.re_parameterize = ReParameterize()

        self.cfg = cfg
        self.enc_conv_1 = nn.Sequential(
            nn.Conv2d(1, cfg["enc_conv_1_ch"], (cfg["conv_ker"],cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]),get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["enc_conv_1_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.enc_conv_2 = nn.Sequential(
            nn.Conv2d(cfg["enc_conv_1_ch"], cfg["enc_conv_2_ch"], (cfg["conv_ker"],cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]),get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["enc_conv_2_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.enc_conv_4 = nn.Sequential(
            nn.Conv2d(cfg["enc_conv_2_ch"], cfg["enc_conv_4_ch"]*2, (cfg["conv_ker"],cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]),get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["enc_conv_4_ch"]*2),
            nn.Dropout(p=cfg["dropRate"])
        )
        self.ds_2_2 = nn.MaxPool2d((cfg["ds_ratio"],cfg["ds_ratio"]))
        self.n_recording_chan_1 = cfg["n_channels"]
        self.n_recording_chan_2 = np.floor((self.n_recording_chan_1+2*get_padding(cfg["ds_ratio"])-cfg["ds_ratio"])/cfg["ds_ratio"]).astype('int')+1
        self.n_recording_chan_3 = np.floor((self.n_recording_chan_2+2*get_padding(cfg["ds_ratio"])-cfg["ds_ratio"])/cfg["ds_ratio"]).astype('int')+1
        self.n_recording_chan_4 = np.floor((self.n_recording_chan_3+2*get_padding(cfg["ds_ratio"])-cfg["ds_ratio"])/cfg["ds_ratio"]).astype('int')+1

        self.spk_length_1 = cfg["spk_length"]
        self.spk_length_2 = np.floor((self.spk_length_1+2*get_padding(cfg["ds_ratio"])-cfg["ds_ratio"])/cfg["ds_ratio"]).astype('int')+1
        self.spk_length_3 = np.floor((self.spk_length_2+2*get_padding(cfg["ds_ratio"])-cfg["ds_ratio"])/cfg["ds_ratio"]).astype('int')+1
        self.spk_length_4 = np.floor((self.spk_length_3+2*get_padding(cfg["ds_ratio"])-cfg["ds_ratio"])/cfg["ds_ratio"]).astype('int')+1
        self.dec_conv_1 = nn.Sequential(
            nn.Conv2d(cfg["dec_conv_1_ch"],1, (cfg["conv_ker"],cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]),get_padding(cfg["conv_ker"])), bias=False),
        )
        self.dec_conv_2 = nn.Sequential(
            nn.Conv2d(cfg["dec_conv_2_ch"], cfg["dec_conv_1_ch"], (cfg["conv_ker"],cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]),get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["dec_conv_1_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )

        self.dec_conv_4 = nn.Sequential(
            nn.Conv2d(cfg["dec_conv_4_ch"], cfg["dec_conv_2_ch"], (cfg["conv_ker"],cfg["conv_ker"]), groups=1,
                      padding=(get_padding(cfg["conv_ker"]),get_padding(cfg["conv_ker"])), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cfg["dec_conv_2_ch"]),
            nn.Dropout(p=cfg["dropRate"])
        )

        self.resx_1 = ResNeXtBottleNeck_2d(cfg["dec_conv_1_ch"], cfg["conv_ker"], cardinality=int(cfg["dec_conv_1_ch"]/cfg["cardinality_factor"]),
                                      dropRate=cfg["dropRate"])
        self.resx_2 = ResNeXtBottleNeck_2d(cfg["dec_conv_2_ch"], cfg["conv_ker"], cardinality=int(cfg["dec_conv_2_ch"]/cfg["cardinality_factor"]),
                                      dropRate=cfg["dropRate"])

        self.resnet_1 = BasicResBlock_2d(cfg["dec_conv_1_ch"], cfg["conv_ker"], n_layers=2, decode=True,
                                    dropRate=cfg["dropRate"])
        self.resnet_2 = BasicResBlock_2d(cfg["dec_conv_2_ch"], cfg["conv_ker"], n_layers=2, decode=True,
                                    dropRate=cfg["dropRate"])

        # move model to GPU
        if torch.cuda.is_available():
            self.cuda()
        # loss
        self.criterion = calc_ori_loss2
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.cfg["learn_rate"],
                                    weight_decay=self.cfg["weight_decay"], amsgrad=True)
        self.unique_labels = []
        self.target_means = []

    def encoder(self, x):
        if x.ndim==3:
            x = x[:,None,:,:,]
        out = self.enc_conv_1(x)
        out = self.resx_1(out)
        out = self.ds_2_2(out)
        out = self.enc_conv_2(out)
        out = self.resx_2(out)
        out = self.ds_2_2(out)
        out = self.enc_conv_4(out)
        mu, log_var = torch.split(out,round(out.shape[1]/2),dim=1)
        return mu, log_var


    def decoder(self, x):
        out = self.dec_conv_4(x)
        out = nn.functional.interpolate(out,size=(self.n_recording_chan_2,self.spk_length_2), mode='bicubic')
        out = self.resnet_2(out)
        out = self.dec_conv_2(out)
        out = nn.functional.interpolate(out,size=(self.n_recording_chan_1,self.spk_length_1), mode='bicubic')
        out = self.resnet_1(out)
        out = self.dec_conv_1(out)
        return out

    def forward(self, x):
        enc_mu, enc_log_var = self.encoder(x)

        z = self.re_parameterize(enc_mu, enc_log_var)
        dec_out = self.decoder(z)
        return dec_out, enc_mu, enc_log_var

    def train_epoch(self, train_loader):
        epoch_loss = 0
        last_batch_portion = 0
        self.train()
        t = time.time()
        # do stuff
        for i_batch, (spikes, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                spikes, labels = spikes.cuda(), labels.cuda()
            self.optimizer.zero_grad()
            # Forward Pass
            dec_out, enc_mu, enc_log_var = self.forward(spikes)
            # Compute loss
            loss = self.criterion(spikes, dec_out, enc_mu, enc_log_var, labels, self.unique_labels, self.target_means)
            # Backward Pass
            loss.backward()
            epoch_loss += loss.cpu().detach().numpy()
            # update weights
            self.optimizer.step()

            batch_portion = round(i_batch * train_loader.batch_size / len(train_loader) * 100)
            if batch_portion % 20 == 0 and batch_portion != last_batch_portion:
                elapsed = time.time() - t
                t = time.time()
                print("finish {}% of epoch: loss is {:.3E}, time passed {:.1f} sec".format(batch_portion,
                                                                                           epoch_loss / (
                                                                                                   i_batch * train_loader.batch_size),
                                                                                           elapsed))
                last_batch_portion = batch_portion

        train_loader.reset()
        return epoch_loss / len(train_loader)

    def train_data_loader(self, data_loader):
        n_epochs = self.cfg["n_epochs"]
        loss_array = np.zeros((n_epochs, 1))

        for i_epoch in range(n_epochs):
            epoch_loss = self.train_epoch(data_loader)
            loss_array[i_epoch] = epoch_loss
            if i_epoch % 1 == 0:
                print("epoch {}: loss is {:.3E}".format(i_epoch, epoch_loss))
        return loss_array

    def visual_eval_model(self, data_loader, n_spikes):
        with torch.no_grad():
            self.eval()
            sampler = np.random.permutation(len(data_loader))
            for i_spike in range(n_spikes):
                spike, labels = data_loader.get_spike(sampler[i_spike])
                spike = spike[None, :, :]
                if torch.cuda.is_available():
                    spike, labels = spike.cuda(), labels.cuda()
                recon_spike = self.forward(spike)[0]
                spike = spike.cpu().detach().numpy().squeeze(axis=0)
                recon_spike = recon_spike.cpu().detach().numpy().squeeze(axis=0)
                show_two_spikes(spike, recon_spike.squeeze())

    def forward_encoder(self, train_loader, n_spikes):
        with torch.no_grad():
            self.eval()
            for i_batch, (spikes, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    spikes = spikes.cuda()
                curr_enc_mu = self.encoder(spikes)[0].cpu().detach().numpy()
                curr_label = labels.detach().numpy()
                if i_batch == 0:
                    all_spk = spikes.cpu().detach().numpy()
                    enc_mu = curr_enc_mu
                    all_label = curr_label
                else:
                    enc_mu = np.concatenate((enc_mu, curr_enc_mu), axis=0)
                    all_label = np.concatenate((all_label, curr_label), axis=0)
                    all_spk = np.concatenate((all_spk, spikes.cpu().detach().numpy()), axis=0)
                if enc_mu.shape[0] > n_spikes:
                    break
            train_loader.reset()
            return enc_mu, all_label, all_spk

    def calc_means(self, data_loader):
        all_sample = torch.tensor(())
        all_label = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.cuda()
                mu, var = self.encoder(images)
                mu = mu.cpu()
                all_sample = torch.cat((all_sample, mu), 0)
                all_label.append(labels.numpy().flatten())
        all_label = np.hstack(all_label)
        all_sample = all_sample.numpy()
        # 2D feature space of size [#spikes X #2d_channels X #recording_channels (at encoder center) X #time_samples(at encoder center)]
        all_sample = all_sample.reshape(all_sample.shape[0],-1)
        unique_labelse = np.sort(np.unique(all_label))
        means = np.zeros([len(unique_labelse), all_sample.shape[1]])
        for i_label in range(len(unique_labelse)):
            means[i_label, :] = np.mean(all_sample[all_label == unique_labelse[i_label], :], axis=0)
        return unique_labelse, means

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg
        }, path)

    @classmethod
    def load_vae_model(cls, path):
        checkpoint = torch.load(path)
        vea = cls(checkpoint['cfg'])
        vea.load_state_dict(checkpoint['model_state_dict'])
        vea.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return vea

# from dataHandle import SpikeDataLoader
# import os
# batch_size = 1024
# shuffle = True
# file_dirs = ["C:/DL_data"]
# file_clu_names = ["mF105_10.spk.1"]
# data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)
#
# resnet_cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
#           "enc_conv_1_ch": 32, "enc_conv_2_ch": 64, "enc_conv_3_ch": 128, "enc_conv_4_ch": 8,
#           "dec_conv_1_ch": 32, "dec_conv_2_ch": 64, "dec_conv_3_ch": 128, "dec_conv_4_ch": 8,
#           "conv_ker": 3,
#           "ds_ratio": 2,
#           "cardinality_factor": 8, "dropRate": 0.2, "n_epochs": 15,
#           "learn_rate": 1e-3, "weight_decay": 1e-5}
# n_epochs = 100
# vae_model = resnet_2d_vae(resnet_cfg)
# vae_model.train_data_loader(data_loader)
# vae_model.save_model(os.path.join(os.getcwd(), 'simple_vae.pt'))
