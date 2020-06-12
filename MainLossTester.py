from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob

import os
import torch
from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

"""# VAE class (and sub class)"""


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
        self.enc_hidden = nn.Linear(in_features=np.prod(cfg["input_shape"]), out_features=cfg["hidden_dim"])
        self.enc_activation = nn.Softplus()
        self.enc_mu = nn.Linear(in_features=cfg["hidden_dim"], out_features=cfg["latent_dim"])
        self.enc_log_var = nn.Linear(in_features=cfg["hidden_dim"], out_features=cfg["latent_dim"])

    def forward(self, x):
        enc_hidden = self.enc_activation(self.enc_hidden(x))
        enc_mu = self.enc_mu(enc_hidden)
        enc_log_var = self.enc_log_var(enc_hidden)
        return enc_mu, enc_log_var


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dec_hidden = nn.Linear(in_features=cfg["latent_dim"], out_features=cfg["hidden_dim"])
        self.dec_hidden_activation = nn.Softplus()
        self.dec_out = nn.Linear(in_features=cfg["hidden_dim"], out_features=np.prod(cfg["input_shape"]))
        self.dec_out_activation = nn.Sigmoid()

    def forward(self, z):
        dec_hidden = self.dec_hidden_activation(self.dec_hidden(z))
        dec_out = self.dec_out_activation(self.dec_out(dec_hidden))
        return dec_out


def calc_vae_loss(x, x_recon, enc_mu, enc_log_var, y, unique_labels, target_means):
    reconstruction_loss = F.mse_loss(x, x_recon, reduction='sum')
    # Loss = nn.BCELoss(reduction='sum')
    # reconstruction_loss = Loss(x_recon, x)
    # reconstruction_loss = torch.sum(-x * x_recon.log())
    kld_loss = -0.5 * torch.sum(1 + enc_log_var - enc_mu ** 2 - torch.exp(enc_log_var))
    # print('KL loss =',kld_loss )
    # print('reconstruction_loss =',reconstruction_loss )
    return kld_loss + reconstruction_loss

def calc_recon_loss(x, x_recon, enc_mu, enc_log_var, y, unique_labels, target_means):
    reconstruction_loss = F.mse_loss(x, x_recon, reduction='sum')
    # Loss = nn.BCELoss(reduction='sum')
    # reconstruction_loss = Loss(x_recon, x)
    # reconstruction_loss = torch.sum(-x * x_recon.log())
    # kld_loss = -0.5 * torch.sum(1 + enc_log_var - enc_mu ** 2 - torch.exp(enc_log_var))
    # print('KL loss =',kld_loss )
    # print('reconstruction_loss =',reconstruction_loss )
    return reconstruction_loss


def calc_ori_loss(x, x_recon, enc_mu, enc_log_var, y, unique_labels, target_means):
    reconstruction_loss = F.mse_loss(x, x_recon, reduction='sum')

    # Loss = nn.BCELoss(reduction='sum')
    # reconstruction_loss = Loss(x_recon, x)
    # reconstruction_loss = torch.sum(-x * x_recon.log())+
    kld_loss = 0
    classes = np.unique(y.cpu())
    for i_class in range(len(classes)):
        class_mus = enc_mu[y == classes[i_class], :]
        class_log_vars = enc_log_var[y == classes[i_class], :]

        n_samples = sum(y == classes[i_class])
        class_mu = torch.mean(class_mus, axis=0)
        # class_mu.requires_grad = False
        class_var = (torch.sum(torch.exp(class_log_vars), axis=0) / n_samples ** 2)
        # class_var.requires_grad = False
        kld_loss += .5 * (torch.sum(torch.log(class_var) -
                                    class_log_vars + (torch.exp(class_log_vars) / class_var) +
                                    (class_mus - class_mu) ** 2 / class_var -
                                    1))
    # kld_loss = -0.5 * torch.sum(1 + enc_log_var - enc_mu**2 - torch.exp(enc_log_var))
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
        class_mus = enc_mu[y == classes[i_class], :]
        class_log_vars = enc_log_var[y == classes[i_class], :]

        n_samples = sum(y == classes[i_class])
        class_mu = torch.tensor(target_means[unique_labels[i_class]][:]).cuda()
        class_var = torch.ones(class_mu.shape).cuda()
        kld_loss += .5 * (torch.sum(torch.log(class_var) -
                                    class_log_vars + (torch.exp(class_log_vars) / class_var) +
                                    (class_mus - class_mu) ** 2 / class_var -
                                    1))
    # kld_loss = -0.5 * torch.sum(1 + enc_log_var - enc_mu**2 - torch.exp(enc_log_var))
    # print('KL loss =',kld_loss )
    # print('reconstruction_loss =',reconstruction_loss )
    return kld_loss + reconstruction_loss


class Vae(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.encoder.cuda()
        self.re_parameterize = ReParameterize()
        self.decoder = Decoder(cfg)
        self.decoder.cuda()
        if cfg['loss_idx'] == 1:
            self.criterion = calc_vae_loss
        elif cfg['loss_idx'] == 2:
            self.criterion = calc_ori_loss
        elif cfg['loss_idx'] == 3:
            self.criterion = calc_ori_loss2
        elif cfg['loss_idx'] == 4:
            self.criterion = calc_recon_loss
        self.unique_labels = []
        self.target_means = []

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),
                                                  lr=cfg["learning_rate"],
                                                  weight_decay=cfg["decay"])
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),
                                                  lr=cfg["learning_rate"],
                                                  weight_decay=cfg["decay"])
        self.cfg = cfg

    def forward(self, x):
        enc_mu, enc_log_var = self.encoder(x)
        z = self.re_parameterize(enc_mu, enc_log_var)
        dec_out = self.decoder(z)
        return dec_out, enc_mu, enc_log_var

    def train_epoch(self, train_loader):
        epoch_loss = 0
        self.encoder.train()
        self.decoder.train()
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            images = flatten(images)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Forward Pass
            dec_out, enc_mu, enc_log_var = self.forward(images)
            # Compute loss
            loss = self.criterion(images, dec_out, enc_mu, enc_log_var, labels, self.unique_labels, self.target_means)
            # Backward Pass
            loss.backward()
            epoch_loss += loss.cpu().detach().numpy()
            # update weights
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        return epoch_loss / len(train_loader)

    def train(self, data_loader, n_epochs):
        loss_array = np.zeros((n_epochs, 1))

        for i_epoch in range(n_epochs):
            epoch_loss = self.train_epoch(vae_train_loader)
            loss_array[i_epoch] = epoch_loss
            if i_epoch % 1 == 0:
                print("epoch {}: loss is {}".format(i_epoch, epoch_loss))
        # plt.plot(loss_array)
        # plt.show()
        return loss_array

    def visual_eval_model(self, data_loader):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            for images, labels in data_loader:
                shape = images.size()
                if torch.cuda.is_available():
                    images, labels = Variable(images.cuda()), Variable(labels.cuda())
                else:
                    images, labels = Variable(images), Variable(labels)
                images_f = flatten(images)
                dec_out = self.forward(images_f)[0]
                out_images = un_flatten(dec_out, shape)
                n = 10
                digit_size = 28
                figure_re = np.zeros((digit_size * n, digit_size * n))
                figure_orij = np.zeros((digit_size * n, digit_size * n))

                for j in range(n):
                    for i in range(n):
                        d_x = i * digit_size
                        d_y = j * digit_size
                        figure_re[d_x:d_x + digit_size, d_y:d_y + digit_size] = out_images[j * 10 + i, 0, :, :].cpu()
                        figure_orij[d_x:d_x + digit_size, d_y:d_y + digit_size] = images[j * 10 + i, 0, :, :].cpu()
                plt.figure(1, figsize=(10, 10))
                plt.title("reconstructed")
                plt.imshow(figure_re, cmap='gray_r')
                plt.figure(2, figsize=(10, 10))
                plt.title("original")
                plt.imshow(figure_orij, cmap='gray_r')

                plt.show()
                return

    def calc_means(self, data_loader):
        all_sample = torch.tensor(())
        all_label = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = flatten(images).cuda()
                mu, var = self.encoder(images)
                mu = mu.cpu()
                var = var.cpu()
                for i_sample in range(1):
                    sample = self.re_parameterize(mu, var / 1e10)
                    all_sample = torch.cat((all_sample, sample), 0)
                    all_label.append(labels)
        all_label = np.asarray(all_label)
        all_sample = all_sample.numpy()
        unique_labelse = np.sort(np.unique(all_label))
        means = np.zeros([len(unique_labelse), all_sample.shape[1]])
        for i_label in range(len(unique_labelse)):
            means[i_label, :] = np.mean(all_sample[all_label == unique_labelse[i_label], :], axis=0)
        return unique_labelse, means

    def plot_pca(self, data_loader,name):
        all_sample = torch.tensor(())
        all_label = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = flatten(images).cuda()
                mu, var = self.encoder(images)
                mu = mu.cpu()
                var = var.cpu()
                for i_sample in range(1):
                    sample = self.re_parameterize(mu, var / 1e10)
                    all_sample = torch.cat((all_sample, sample), 0)
                    all_label.append(labels)

        pca = PCA(n_components=2)
        # pca.fit(all_sample)
        projected = pca.fit_transform(all_sample)
        fig = plt.figure()
        ax = plt.subplot(111)

        ax.scatter(projected[:, 0], projected[:, 1],
                    c=all_label, edgecolor='none', alpha=0.5, cmap='jet')
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title(name)
        fig.savefig(name + '.png')
        plt.clf()
        # plt.show()

    def save_model(self, path):
        torch.save({
            'encoder_model_state_dict': self.encoder.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'decoder_model_state_dict': self.decoder.state_dict(),
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'cfg': self.cfg
        }, path)

    @classmethod
    def load_vae_model(cls, path):
        checkpoint = torch.load(path)
        vea = cls(checkpoint['cfg'])
        vea.encoder.load_state_dict(checkpoint['encoder_model_state_dict'])
        vea.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        vea.decoder.load_state_dict(checkpoint['decoder_model_state_dict'])
        vea.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        return vea


"""# SVM  Classifier class"""


class SvmClassifier(object):
    def __init__(self, n_latent_samples, encoder):
        self.classifier = SVC(gamma='auto')
        self.n_latent_samples = n_latent_samples
        self.encoder = encoder
        self.encoder.cpu()
        self.encoder.eval()
        self.re_parameterize = ReParameterize()

    def train(self, labeled_train_loader):
        all_sample = torch.tensor(())
        all_label = torch.tensor((), dtype=torch.long)

        with torch.no_grad():
            for images, labels in labeled_train_loader:
                images = flatten(images)
                mu, var = self.encoder(images)
                for i_sample in range(self.n_latent_samples):
                    sample = self.re_parameterize(mu, var)
                    all_sample = torch.cat((all_sample, sample), 0)
                    all_label = torch.cat((all_label, labels), 0)

        all_sample = all_sample.numpy()
        all_label = all_label.numpy()
        self.classifier.fit(all_sample, all_label)

    def test(self, test_loader):
        all_mu = torch.tensor(())
        all_label = torch.tensor((), dtype=torch.long)
        with torch.no_grad():
            for images, labels in test_loader:
                images = flatten(images)
                mu, _ = self.encoder(images)
                all_mu = torch.cat((all_mu, mu), 0)
                all_label = torch.cat((all_label, labels), 0)
        test_pred = self.classifier.predict(all_mu)
        acc = accuracy_score(test_pred, all_label)
        return acc

    def save_model(self, path):
        filehandler = open(path, "wb")
        pickle.dump(self, filehandler)
        filehandler.close()

    @classmethod
    def load_model(cls, path):
        file = open(path, 'rb')
        classifier = pickle.load(file)
        file.close()
        return classifier


"""# load data"""
path = r"C:\git\SpikeSortingVAE\loss_tester"
torch.manual_seed(0)
data_root = 'FasionMNIST_data'
# train
train_set = datasets.FashionMNIST(os.path.join(path, data_root), download=True, train=True,
                                  transform=transforms.ToTensor())
vae_train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024, shuffle=True)
# test
test_set = datasets.FashionMNIST(os.path.join(path, data_root), download=True, train=False,
                                 transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

"""training base vae with  regular vae loos"""
do_recon_only_train = False
if do_recon_only_train:
    torch.manual_seed(0)
    cfg = {"latent_dim": 20,
           "hidden_dim": 600,
           "input_shape": vae_train_loader.dataset.data[1, :, :].shape,
           "learning_rate": 0.0003,
           "decay": 0.00,
           "loss_idx": 4,
           }
    n_epochs = 200
    vae_model = Vae(cfg)
    vae_model.train(vae_train_loader, n_epochs)
    vae_model.visual_eval_model(vae_train_loader)
    vae_model.save_model(
        os.path.join(path, 'vae_model_only_recon_LT{}_LR{}.pt'.format(cfg['latent_dim'], cfg['learning_rate'])))
    vae_model.plot_pca(test_loader,'vae_model_only_recon_LT{}_LR{}.pt'.format(cfg['latent_dim'], cfg['learning_rate']))

"""training base vae with  regular vae loos"""
do_base_train = False
if do_base_train:
    torch.manual_seed(0)
    cfg = {"latent_dim": 20,
           "hidden_dim": 600,
           "input_shape": vae_train_loader.dataset.data[1, :, :].shape,
           "learning_rate": 0.0003,
           "decay": 0.00,
           "loss_idx": 1,
           }
    n_epochs = 200
    vae_model = Vae(cfg)
    vae_model.train(vae_train_loader, n_epochs)
    vae_model.visual_eval_model(vae_train_loader)
    vae_model.save_model(
        os.path.join(path, 'vae_model_normal_L{}_LR{}.pt'.format(cfg['latent_dim'], cfg['learning_rate'])))
    vae_model.plot_pca(test_loader, 'vae_model_normal_L{}_LR{}.pt'.format(cfg['latent_dim'], cfg['learning_rate']))
""" grid train of ori loss2 """
do_grid_loss_train = False
if do_grid_loss_train:
    # load trained vae
    torch.manual_seed(0)
    vae_model = Vae.load_vae_model(os.path.join(path, 'vae_model_normal_LT20_LR0.0003.pt'))
    # set criterion
    vae_model.criterion = calc_ori_loss2
    unique_labels, target_means = vae_model.calc_means(train_set)
    # define grid
    factors = [1, 2, 5, 10]
    LRs = [3e-3, 3e-4, 3e-5]
    # train
    for fact in factors:
        for lr in LRs:
            vae_model = Vae.load_vae_model(os.path.join(path, 'vae_model_normal_LT20_LR0.0003.pt'))
            vae_model.criterion = calc_ori_loss2

            cur_target_means = target_means * fact
            vae_model.encoder_optimizer = torch.optim.Adam(vae_model.encoder.parameters(),
                                                           lr=lr,
                                                           weight_decay=cfg["decay"])
            vae_model.decoder_optimizer = torch.optim.Adam(vae_model.decoder.parameters(),
                                                           lr=lr,
                                                           weight_decay=cfg["decay"])

            vae_model.unique_labels = unique_labels
            vae_model.target_means = cur_target_means
            vae_model.train(vae_train_loader, 100)
            vae_model.save_model(os.path.join(path, 'vae_model_ori2_Fact{}_LR{}.pt'.format(fact, lr)))

"""" plot PCA for all models """

do_plot_pca = True

if do_plot_pca:
    vae_model = Vae.load_vae_model(os.path.join(path, 'vae_model_normal_LT20_LR0.0003.pt'))
    vae_model.plot_pca(test_set,'base')
    vae_model = Vae.load_vae_model(os.path.join(path, 'vae_model_only_recon_LT20_LR0.0003.pt'))
    vae_model.plot_pca(test_set,'recon only')
    model_list = glob.glob(r'C:\git\SpikeSortingVAE\loss_tester\vae_model_ori2*.pt')
    for model in model_list:
        vae_model = Vae.load_vae_model(model)
        vae_model.plot_pca(test_set,os.path.split(model)[1])
"""# train and save SVMs"""

labeled_data_sizes = [100]
for labeled_data_size in labeled_data_sizes:
    n_images = len(train_set)
    train_idx, _ = train_test_split(
        np.arange(n_images),
        test_size=(n_images - labeled_data_size) / n_images,
        shuffle=True,
        stratify=train_set.targets)
    subset_train_set = torch.utils.data.Subset(train_set, train_idx)
    svn_train_loader = torch.utils.data.DataLoader(subset_train_set, batch_size=100, shuffle=True)
    classifier = SvmClassifier(50, vae_model.encoder)
    classifier.train(svn_train_loader)
    acc = classifier.test(test_loader)
    print("using {} labeled images accuracy was {}".format(labeled_data_size, acc))
    classifier.save_model(os.path.join(path, "SVM_model_{}_labels.pt".format(labeled_data_size)))

"""# Test SVMs"""

labeled_data_sizes = [100]
for labeled_data_size in labeled_data_sizes:
    classifier = SvmClassifier.load_model(os.path.join(path, "SVM_model_{}_labels.pt".format(labeled_data_size)))
    acc = classifier.test(test_loader)
    print("using {} labeled images accuracy was {}".format(labeled_data_size, acc))
