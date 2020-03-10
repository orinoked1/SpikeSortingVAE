from dataHandle import SpikeDataLoader
from autoEncoder import Vae
from ClassificationTester import ClassificationTester
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle


do_train = True
if do_train:
    batch_size = 128
    shuffle = True
    file_dirs = [os.path.join(os.getcwd(), 'example_data')]
    file_clu_names = ["mF105_10.spk.2" ]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)
    for latent_dim in [27]:
        for learn_rate in [1e-3, 1e-4]:  # default 1e-3
            for weight_decay in [1e-4]:  # default 1e-4
                for shuffle_channels in [False]:  # default False
                    for drop_rate in [0.2]:  # default 0.2
                        cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
                               "conv1_ch": 256, "conv2_ch": 16, "latent_dim": latent_dim,
                               "conv0_ker": 1, "conv1_ker": 3, "conv2_ker": 1,
                               "ds_ratio_1": 2, "ds_ratio_2": 2,
                               "cardinality": 32, "dropRate": drop_rate, "n_epochs": 1,
                               "learn_rate": learn_rate, "weight_decay": weight_decay,
                               "shuffle_channels": shuffle_channels}
                        cfg["ds_ratio_tot"] = cfg["ds_ratio_2"] * cfg["ds_ratio_1"]
                        torch.manual_seed(0)
                        np.random.seed(0)
                        # training
                        vae_model = Vae(cfg)
                        loss_array = vae_model.train_data_loader(data_loader)
                        plt.plot(loss_array)
                        plt.savefig(os.path.join(os.getcwd(),'vae_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.png'.format(latent_dim, learn_rate, weight_decay, shuffle_channels, drop_rate)))
                        plt.clf()
                        vae_model.save_model(os.path.join(os.getcwd(),'vae_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.pt'.format(latent_dim, learn_rate, weight_decay, shuffle_channels, drop_rate)))

do_search = True
if do_search:
    max_acc = 0
    model_list = glob.glob(os.path.join(os.getcwd(), 'vae_LD27*.pt'))
    for i_model in range(len(model_list)):
        file_dirs = [os.path.join(os.getcwd(), 'example_data')]
        file_clu_names = ["mF105_10.spk.2", ]
        torch.manual_seed(0)
        np.random.seed(0)
        data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=500, shuffle=False)
        vae_model = Vae.load_vae_model(model_list[i_model])
        feat, classes, spk_data = vae_model.forward_encoder(data_loader, 1e5)

        unique_classes, class_counts = np.unique(classes, return_counts=True)
        small_labels = unique_classes[class_counts < 70]
        spk_data = spk_data[(classes != small_labels).all(axis=1), :, :]
        feat = feat[(classes != small_labels).all(axis=1), :]
        classes = classes[(classes != small_labels).all(axis=1)]

        classifier2 = ClassificationTester(feat, classes, use_pca=False)
        print('model {} had acc of {}'.format(model_list[i_model], classifier2.gmm_acc))
        if classifier2.gmm_acc > max_acc:
            max_acc = classifier2.gmm_acc
            best_model = i_model


do_full_eval = True
if do_full_eval:
    file_dirs = [os.path.join(os.getcwd(), 'example_data')]
    file_clu_names = ["mF105_10.spk.2"]
    torch.manual_seed(0)
    np.random.seed(0)
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=500, shuffle=False)
    vae_model = Vae.load_vae_model(model_list[i_model])
    feat, classes, spk_data = vae_model.forward_encoder(data_loader, 1e6)

    unique_classes, class_counts = np.unique(classes, return_counts=True)
    small_labels = unique_classes[class_counts < 70]
    spk_data = spk_data[(classes != small_labels).all(axis=1), :, :]
    feat = feat[(classes != small_labels).all(axis=1), :]
    classes = classes[(classes != small_labels).all(axis=1)]

    classification_tester_pca = ClassificationTester(spk_data, classes, use_pca=False, n_init=10, max_iter=100)
    classification_tester_vae = ClassificationTester(feat, classes, use_pca=False, n_init=10, max_iter=100)
    with open('classification_tester_pca_200k.pt', 'wb') as file:
        pickle.dump(classification_tester_pca, file)
    with open('classification_tester_vae_200k.pt', 'wb') as file:
        pickle.dump(classification_tester_vae,file)

create_figures = True
if create_figures:
    with open('classification_tester_pca_200k.pt', 'rb') as file:
        classification_tester_pca = pickle.load(file)
    with open('classification_tester_vae_200k.pt', 'rb') as file:
        classification_tester_vae = pickle.load(file)

    classification_tester_pca.plot_2d_pca()
    classification_tester_vae.plot_2d_pca()
    classification_tester_pca.plot_2d_pca_mat(classification_tester_vae.gmm_pairwise_acc)
    classification_tester_vae.plot_2d_pca_mat(classification_tester_pca.gmm_pairwise_acc)
