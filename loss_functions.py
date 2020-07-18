import torch
from torch import nn
import torch.nn.functional as F

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
    enc_mu = enc_mu.flatten(start_dim=1)
    enc_log_var = enc_log_var.flatten(start_dim=1)
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
