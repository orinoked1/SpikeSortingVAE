import torch
from torch import nn
import numpy as np
import time
from dataHandle import show_two_spikes
from utils import ReParameterize
from loss_functions import calc_ori_loss2




class AutoEncoderBaseClass(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoderBaseClass, self).__init__()

        self.re_parameterize = ReParameterize()
        self.cfg = cfg

        # loss
        self.criterion = calc_ori_loss2

        self.unique_labels = []
        self.target_means = []

    def encoder(self, x):
        return x, x

    def decoder(self, x):
        return x

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

    def forward_encoder(self, train_loader, n_spikes, return_mu=True, return_spk=True):
        with torch.no_grad():
            self.eval()
            for i_batch, (spikes, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    spikes = spikes.cuda()
                curr_enc_mu = self.encoder(spikes)[0].cpu().detach().numpy()
                curr_label = labels.detach().numpy()
                curr_idxs = train_loader.sampler[train_loader._index - train_loader.batch_size:train_loader._index]

                if i_batch == 0:
                    all_spk_idx = curr_idxs
                    enc_mu = curr_enc_mu
                    all_label = curr_label
                else:
                    if return_mu:
                        enc_mu = np.concatenate((enc_mu, curr_enc_mu), axis=0)
                    all_label = np.concatenate((all_label, curr_label), axis=0)
                    if return_spk:
                        all_spk_idx = np.concatenate((all_spk_idx, curr_idxs), axis=0)
                if enc_mu.shape[0] > n_spikes:
                    break
            train_loader.reset()
            return enc_mu, all_label, all_spk_idx

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
        # 2D feature space of size [#spikes X #2d_channels X #recording_channels (at encoder center) X #time_samples(at encoder center)]
        all_sample = all_sample.reshape(all_sample.shape[0], -1)
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
