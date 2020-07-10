from dataHandle import SpikeDataLoader
from autoEncoder import Vae
from ClassificationTester import ClassificationTester
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle


do_train = False
if do_train:
    batch_size = 128
    shuffle = True
    file_dirs = [os.path.join(os.getcwd(), 'example_data')]
    file_clu_names = ["mF105_10.spk.2" ]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)
    for latent_dim in [27]:
        for learn_rate in [1e-3, 1e-4]:  # default 1e-3
            for weight_decay in [1e-4,1e-5]:  # default 1e-4
                for shuffle_channels in [False,True]:  # default False
                    for drop_rate in [0.2,0.5]:  # default 0.2
                        cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
                               "conv1_ch": 256, "conv2_ch": 16, "latent_dim": latent_dim,
                               "conv0_ker": 1, "conv1_ker": 3, "conv2_ker": 1,
                               "ds_ratio_1": 2, "ds_ratio_2": 2,
                               "cardinality": 32, "dropRate": drop_rate, "n_epochs": 15,
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
do_search = False
if do_search:
    max_acc = 0
    model_list = glob.glob(os.path.join(os.getcwd(), 'vae_LD27*.pt'))
    for i_model in range(len(model_list)):
        file_dirs = [os.path.join(os.getcwd(), 'example_data')]
        file_clu_names = ["mF105_10.spk.2", ]
        torch.manual_seed(0)
        np.random.seed(0)
        data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
        vae_model = Vae.load_vae_model(model_list[i_model])
        feat, classes, all_spk_idx = vae_model.forward_encoder(data_loader, 1e5)
        unique_classes, class_counts = np.unique(classes, return_counts=True)
        small_labels = unique_classes[class_counts < 70]
        feat = feat[(classes != small_labels).all(axis=1), :]
        classes = classes[(classes != small_labels).all(axis=1)]

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model],good_clusters=data_loader.good_clusters)
        print('model {} had acc of {}'.format(model_list[i_model], classifier2.gmm_acc))
        if classifier2.gmm_acc > max_acc:
            max_acc = classifier2.gmm_acc
            best_model = i_model
    spk_data = data_loader.spk_data[all_spk_idx]
    spk_data = spk_data[(classes != small_labels).all(axis=1), :, :]

    classifier2 = ClassificationTester(spk_data, classes, use_pca=False, name='pca',good_clusters=data_loader.good_clusters)

do_full_eval = False
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

create_figures = False
if create_figures:
    with open('classification_tester_pca_200k.pt', 'rb') as file:
        classification_tester_pca = pickle.load(file)
    with open('classification_tester_vae_200k.pt', 'rb') as file:
        classification_tester_vae = pickle.load(file)

    classification_tester_pca.plot_2d_pca()
    classification_tester_vae.plot_2d_pca()
    classification_tester_pca.plot_2d_pca_mat(classification_tester_vae.gmm_pairwise_acc)
    classification_tester_vae.plot_2d_pca_mat(classification_tester_pca.gmm_pairwise_acc)

do_2_stage_train = True
if do_2_stage_train:

    batch_size = 2048
    shuffle = True
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.1"]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)

    cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
           "conv1_ch": 256, "conv2_ch": 16, "latent_dim": 27,
           "conv0_ker": 1, "conv1_ker": 3, "conv2_ker": 1,
           "ds_ratio_1": 2, "ds_ratio_2": 2,
           "cardinality": 32, "dropRate": 0.2, "n_epochs": 15,
           "learn_rate": 1e-3, "weight_decay": 1e-5,
           "shuffle_channels": False}
    cfg["ds_ratio_tot"] = cfg["ds_ratio_2"] * cfg["ds_ratio_1"]
    torch.manual_seed(0)
    np.random.seed(0)
    # training
    # vae_model = Vae(cfg)
    # loss_array = vae_model.train_data_loader(data_loader)
    # plt.plot(loss_array)
    # plt.savefig(os.path.join(os.getcwd(),
    #                          'vaeStage1_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.png'.format(cfg["latent_dim"],
    #                                                                                cfg["learn_rate"],
    #                                                                                cfg["weight_decay"],
    #                                                                                cfg["shuffle_channels"],
    #                                                                                cfg["dropRate"])))
    # plt.clf()
    # vae_model.save_model(os.path.join(os.getcwd(),
    #                                   'vaeStage1_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.pt'.format(cfg["latent_dim"],
    #                                                                                        cfg["learn_rate"],
    #                                                                                        cfg["weight_decay"],
    #                                                                                        cfg["shuffle_channels"],
    #                                                                                        cfg["dropRate"])))
    vae_model = Vae.load_vae_model(os.path.join(os.getcwd(),'vaeStage1_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.pt'.format(cfg["latent_dim"],
                                                                                           cfg["learn_rate"],
                                                                                           cfg["weight_decay"],
                                                                                           cfg["shuffle_channels"],
                                                                                           cfg["dropRate"])))
    unique_labels, target_means = vae_model.calc_means(data_loader)
    factors = [1, 2, 5, 10]
    LRs = [3e-3, 3e-4,3e-5]
    # train
    for fact in factors:
        for lr in LRs:

            vae_model = Vae.load_vae_model(os.path.join(os.getcwd(), 'vaeStage1_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.pt'.format(cfg["latent_dim"],
                                                                                           cfg["learn_rate"],
                                                                                           cfg["weight_decay"],
                                                                                           cfg["shuffle_channels"],
                                                                                           cfg["dropRate"])))
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
                                     'vaeStage2_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}_F{}.png'.format(cfg["latent_dim"],
                                                                                           lr,
                                                                                           cfg["weight_decay"],
                                                                                           cfg["shuffle_channels"],
                                                                                           cfg["dropRate"],
                                                                                           fact)))
            plt.clf()
            vae_model.save_model(os.path.join(os.getcwd(),
                                              'vaeStage2_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}_F{}.pt'.format(
                                                  cfg["latent_dim"],
                                                  lr,
                                                  cfg["weight_decay"],
                                                  cfg["shuffle_channels"],
                                                  cfg["dropRate"],
                                                  fact)))
            del vae_model
