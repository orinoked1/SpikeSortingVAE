import torch
from torch import nn
import numpy as np
import time
import torch.nn.functional as F
from utils import make_layers, get_padding
from reslib import ResNeXtBottleNeck, BasicResBlock
from torch import optim
from dataHandle import show_two_spikes


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


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(int(np.prod(cfg["input_shape"])))
        self.enc_hidden = nn.Linear(in_features=int(np.prod(cfg["input_shape"])),
                                    out_features=cfg["hidden_dim"])
        self.hidden_bn = nn.BatchNorm1d(cfg["hidden_dim"])
        self.enc_activation = nn.Softplus()
        self.enc_mu = nn.Linear(in_features=cfg["hidden_dim"],
                                out_features=cfg["latent_dim"])
        self.enc_log_var = nn.Linear(in_features=cfg["hidden_dim"],
                                     out_features=cfg["latent_dim"])

    def forward(self, x):
        x = self.input_bn(x)
        enc_hidden = self.hidden_bn(self.enc_activation(self.enc_hidden(x)))
        enc_mu = self.enc_mu(enc_hidden)
        enc_log_var = self.enc_log_var(enc_hidden)
        return enc_mu, enc_log_var


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dec_hidden = nn.Linear(in_features=cfg["latent_dim"],
                                    out_features=cfg["hidden_dim"])
        self.hidden_bn = nn.BatchNorm1d(cfg["hidden_dim"])
        self.dec_hidden_activation = nn.Softplus()
        self.dec_out = nn.Linear(in_features=cfg["hidden_dim"],
                                 out_features=int(np.prod(cfg["input_shape"])))
        self.dec_out_activation = nn.Sigmoid()

    def forward(self, z):
        dec_hidden = self.hidden_bn(self.dec_hidden_activation(self.dec_hidden(z)))
        dec_out = self.dec_out_activation(self.dec_out(dec_hidden))
        return dec_out


# def calc_vae_loss(x, x_recon, enc_mu, enc_log_var):
#     Loss = nn.MSELoss(reduction='sum')
#     reconstruction_loss = Loss(x_recon, x)
#     kld_loss = -0.5 * torch.sum(1 + enc_log_var - enc_mu ** 2 - torch.exp(enc_log_var))
#     return kld_loss + reconstruction_loss
def calc_vae_loss(x, x_recon, enc_mu, enc_log_var, y, unique_labels, target_means):
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


class Vae(nn.Module):
    def __init__(self, cfg):
        super(Vae, self).__init__()

        self.re_parameterize = ReParameterize()

        self.cfg = cfg

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
        # loss
        self.criterion = calc_ori_loss2
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
            if self.cfg["shuffle_channels"]:
                spikes = spikes[:, np.random.permutation(self.cfg["n_channels"]), :]
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
                show_two_spikes(spike, recon_spike)

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
# torch.manual_seed(0)
# path = "D:/Drive/DL_project_OD/Code"
# data_root = 'FasionMNIST_data'
# # train
# train_set = datasets.FashionMNIST(os.path.join(path, data_root), download=True, train=True, transform=transforms.ToTensor())
# vae_train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
# # test
# test_set = datasets.FashionMNIST(os.path.join(path, data_root), download=True, train=False, transform=transforms.ToTensor())
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
# torch.manual_seed(0)
# cfg = {"latent_dim": 50,
#        "hidden_dim": 600,
#        "input_shape": vae_train_loader.dataset.data[1, :, :].shape,
#        "learning_rate": 0.0003,
#        "decay": 0.00,
#        }
# n_epochs = 100
# vae_model = Vae(cfg)
# vae_model.train_data_loader(vae_train_loader,n_epochs)
# vae_model.visual_eval_model(vae_train_loader)
# vae_model.save_model(os.path.join(path, 'vae_model_10LD_w_normalization.pt'))
