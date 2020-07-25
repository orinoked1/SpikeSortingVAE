
from dataHandle import SpikeDataLoader
from autoEncoder import Vae
from simple_vae import SimpleVae
from resnet_2d_vae import resnet_2d_vae
from resnet_2d_vae_v2 import resnet_2d_vae_v2
from ClassificationTester import ClassificationTester
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import scipy
import random
import pandas as pd
import datetime

final_res = False
if final_res:
    max_acc = 0
    model_list = glob.glob(os.path.join(r'D:\Drive\DL_project_OD\models\bbb+', 'vaeStage3_LD36_LR3E-05_WD1E-05_SDFalse_DR0.2_F60_FMC64.pt'))
    for i_model in range(len(model_list)):
        file_dirs = ["C:/DL_data"]
        file_clu_names = ["mF105_10.spk.4", ]
        torch.manual_seed(0)
        np.random.seed(0)
        data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
        vae_model = Vae.load_vae_model(model_list[i_model])
        feat, classes, all_spk_idx = vae_model.forward_encoder(data_loader, 1e9)
        unique_classes, class_counts = np.unique(classes, return_counts=True)
        # small_labels = unique_classes[class_counts < 70]
        # feat = feat[(classes != small_labels).all(axis=1), :]
        # classes = classes[(classes != small_labels).all(axis=1)]

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model]+'final',good_clusters=data_loader.good_clusters)
        classifier2.plot_2d_pca(-1)
    model_list = glob.glob(os.path.join(r'D:\Drive\DL_project_OD\models\bbb+',
                                        'vaeStage1_LD36_LR3E-05_WD1E-05_SDFalse_DR0.2_F60_FMC64.pt'))
    for i_model in range(len(model_list)):
        file_dirs = ["C:/DL_data"]
        file_clu_names = ["mF105_10.spk.4", ]
        torch.manual_seed(0)
        np.random.seed(0)
        data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
        vae_model = Vae.load_vae_model(model_list[i_model])
        feat, classes, all_spk_idx = vae_model.forward_encoder(data_loader, 1e9)
        unique_classes, class_counts = np.unique(classes, return_counts=True)
        # small_labels = unique_classes[class_counts < 70]
        # feat = feat[(classes != small_labels).all(axis=1), :]
        # classes = classes[(classes != small_labels).all(axis=1)]

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model] + 'final',
                                           good_clusters=data_loader.good_clusters)
        classifier2.plot_2d_pca(-1)

    fet_1 = np.load(r'C:\DL_data\mF105_10.fet.4.npy')
    fet_1 = fet_1[:,4:36]
    clu_data = scipy.io.loadmat(r'C:\DL_data\mF105_10.clu.4.mat')
    clu = clu_data['clu']
    clu = clu[1:]
    full_map_table = np.load( r'C:\DL_data\mapmF105_10.npy')
    good_clusters = full_map_table[full_map_table[:, 1] == 4, 2]
    fet_1 = fet_1[clu.squeeze() > 1, :]
    clu = clu[clu.squeeze() > 1, :]
    classifier2 = ClassificationTester(fet_1, clu, use_pca=False, name="mF105_10.fet.4", good_clusters=good_clusters)
    classifier2.plot_2d_pca(-1)

final_acc = True
if final_acc:
    data_idx = np.genfromtxt(r'C:\git\SpikeSortingVAE\25.07 -PCA- 500000 sampling index.csv', delimiter=',')

    max_acc = 0
    model_list = glob.glob(os.path.join(r'D:\Drive\DL_project_OD\models\bbb+',
                                        'vaeStage3_LD36_LR3E-05_WD1E-05_SDFalse_DR0.2_F60_FMC64.pt'))
    for i_model in range(len(model_list)):
        file_dirs = ["C:/DL_data"]
        file_clu_names = ["mF105_10.spk.4", ]
        torch.manual_seed(0)
        np.random.seed(0)

        data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
        vae_model = Vae.load_vae_model(model_list[i_model])
        feat, classes, all_spk_idx = vae_model.forward_encoder(data_loader, 1e9)

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model] + 'final',
                                           good_clusters=data_loader.good_clusters)
    model_list = glob.glob(os.path.join(r'D:\Drive\DL_project_OD\models\bbb+',
                                        'vaeStage1_LD36_LR3E-05_WD1E-05_SDFalse_DR0.2_F60_FMC64.pt'))
    for i_model in range(len(model_list)):
        file_dirs = ["C:/DL_data"]
        file_clu_names = ["mF105_10.spk.4", ]
        torch.manual_seed(0)
        np.random.seed(0)
        data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
        vae_model = Vae.load_vae_model(model_list[i_model])
        feat, classes, all_spk_idx = vae_model.forward_encoder(data_loader, 1e9)

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model] + 'final',
                                           good_clusters=data_loader.good_clusters)


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
classify_gt=False
if classify_gt:
    fet_1 = np.load(r'C:\DL_data\mF105_10.fet.2.npy')
    fet_1 = fet_1[:,:36]
    clu_data = scipy.io.loadmat(r'C:\DL_data\mF105_10.clu.2.mat')
    clu = clu_data['clu']
    clu = clu[1:]
    full_map_table = np.load( r'C:\DL_data\mapmF105_10.npy')
    good_clusters = full_map_table[full_map_table[:, 1] == 2, 2]

    classifier2 = ClassificationTester(fet_1, clu, use_pca=False, name="mF105_10.fet.2_all",good_clusters=good_clusters)
    fet_1 = fet_1[clu.squeeze()>1,:]
    clu = clu[clu.squeeze()>1,:]
    classifier2 = ClassificationTester(fet_1, clu, use_pca=False, name="mF105_10.fet.2",good_clusters=good_clusters)
    fet_1 = np.load(r'C:\DL_data\mF105_10.fet.1.npy')
    fet_1 = fet_1[:,:36]
    clu_data = scipy.io.loadmat(r'C:\DL_data\mF105_10.clu.1.mat')
    clu = clu_data['clu']
    clu = clu[1:]
    full_map_table = np.load( r'C:\DL_data\mapmF105_10.npy')
    good_clusters = full_map_table[full_map_table[:, 1] == 2, 2]

    classifier2 = ClassificationTester(fet_1, clu, use_pca=False, name="mF105_10.fet.1_all",good_clusters=good_clusters)
    fet_1 = fet_1[clu.squeeze()>1,:]
    clu = clu[clu.squeeze()>1,:]
    classifier2 = ClassificationTester(fet_1, clu, use_pca=False, name="mF105_10.fet.1",good_clusters=good_clusters)
    a= 1
    
do_search = True
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

do_2_stage_train = False
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
    vae_model = Vae(cfg)
    loss_array = vae_model.train_data_loader(data_loader)
    plt.plot(loss_array)
    plt.savefig(os.path.join(os.getcwd(),
                             'vaeStage1_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.png'.format(cfg["latent_dim"],
                                                                                   cfg["learn_rate"],
                                                                                   cfg["weight_decay"],
                                                                                   cfg["shuffle_channels"],
                                                                                   cfg["dropRate"])))
    plt.clf()
    vae_model.save_model(os.path.join(os.getcwd(),
                                      'vaeStage1_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}.pt'.format(cfg["latent_dim"],
                                                                                           cfg["learn_rate"],
                                                                                           cfg["weight_decay"],
                                                                                           cfg["shuffle_channels"],
                                                                                           cfg["dropRate"])))
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

do_train_simple_vae = True
if do_train_simple_vae:
    batch_size = 2048
    shuffle = True
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.1"]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)
    for drop_rate in [0.2, 0.5]:  # default 0.2
        for latent_dim in [8,4,1,2]:
            for learn_rate in [1e-3, 1e-4]:  # default 1e-3
                for weight_decay in [1e-5]:  # default 1e-4
                    simple_cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
                              "enc_conv_1_ch": 32, "enc_conv_2_ch": 64, "enc_conv_3_ch": 128, "enc_conv_4_ch": latent_dim,
                              "dec_conv_1_ch": 32, "dec_conv_2_ch": 64, "dec_conv_3_ch": 128, "dec_conv_4_ch": latent_dim,
                              "conv_ker": 3,
                              "ds_ratio": 2,
                              "cardinality": 32, "dropRate": drop_rate, "n_epochs": 15,
                              "learn_rate": learn_rate, "weight_decay": 1e-5}
                    torch.manual_seed(0)
                    np.random.seed(0)
                    # training
                    vae_model = SimpleVae(simple_cfg)
                    loss_array = vae_model.train_data_loader(data_loader)
                    plt.plot(loss_array)
                    plt.savefig(os.path.join(os.getcwd(),'simple_vae_stage_1_LD{}_LR{:.0E}_WD{:.0E}_DR{:.1f}.png'.format(latent_dim, learn_rate, weight_decay, drop_rate)))
                    plt.clf()
                    vae_model.save_model(os.path.join(os.getcwd(),'simple_vae_stage_1_LD{}_LR{:.0E}_WD{:.0E}_DR{:.1f}.pt'.format(latent_dim, learn_rate, weight_decay, drop_rate)))

do_train_resnet_2d = False
if do_train_resnet_2d:
    batch_size = 2048
    shuffle = True
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.1"]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)
    for drop_rate in [0.2, 0.5]:  # default 0.2
        for latent_dim in [8,4,1,2]:
            for learn_rate in [1e-3, 1e-4]:  # default 1e-3
                for weight_decay in [1e-5]:  # default 1e-4
                    resnet_cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
                              "enc_conv_1_ch": 32, "enc_conv_2_ch": 64, "enc_conv_3_ch": 128, "enc_conv_4_ch": latent_dim,
                              "dec_conv_1_ch": 32, "dec_conv_2_ch": 64, "dec_conv_3_ch": 128, "dec_conv_4_ch": latent_dim,
                              "conv_ker": 3,
                              "ds_ratio": 2,
                              "cardinality_factor": 8, "dropRate": drop_rate, "n_epochs": 15,
                              "learn_rate": learn_rate, "weight_decay": 1e-5}
                    torch.manual_seed(0)
                    np.random.seed(0)
                    # training
                    vae_model = resnet_2d_vae(resnet_cfg)
                    loss_array = vae_model.train_data_loader(data_loader)
                    plt.plot(loss_array)
                    plt.savefig(os.path.join(os.getcwd(),
                                             'resnet_vae_stage_1_LD{}_LR{:.0E}_WD{:.0E}_DR{:.1f}.png'.format(latent_dim,
                                                                                                           learn_rate,
                                                                                                           weight_decay,
                                                                                                           drop_rate)))
                    plt.clf()
                    vae_model.save_model(os.path.join(os.getcwd(),
                                                      'resnet_vae_stage_1_LD{}_LR{:.0E}_WD{:.0E}_DR{:.1f}.pt'.format(latent_dim,
                                                                                                      learn_rate,
                                                                                                      weight_decay,
                                                                                                      drop_rate)))

do_train_resnet_2d_v2 = False
if do_train_resnet_2d_v2:
    batch_size = 2048
    shuffle = True
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.1"]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)
    for drop_rate in [0.2, 0.5]:  # default 0.2
        for latent_dim in [8,4,1,2]:
            for learn_rate in [1e-3, 1e-4]:  # default 1e-3
                for weight_decay in [1e-5]:  # default 1e-4
                    resnet_cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
                              "enc_conv_1_ch": 4, "enc_conv_2_ch": 16, "enc_conv_3_ch": 64, "enc_conv_4_ch": latent_dim,
                              "dec_conv_1_ch": 4, "dec_conv_2_ch": 16, "dec_conv_3_ch": 64, "dec_conv_4_ch": latent_dim,
                              "conv_ker": 3,
                              "ds_ratio": 2,
                              "cardinality_factor": 8, "dropRate": drop_rate, "n_epochs": 15,
                              "learn_rate": learn_rate, "weight_decay": 1e-5}
                    torch.manual_seed(0)
                    np.random.seed(0)
                    # training
                    vae_model = resnet_2d_vae_v2(resnet_cfg)
                    loss_array = vae_model.train_data_loader(data_loader)
                    plt.plot(loss_array)
                    plt.savefig(os.path.join(os.getcwd(),
                                             'resnet_vaeV2_stage_1_LD{}_LR{:.0E}_WD{:.0E}_DR{:.1f}.png'.format(latent_dim,
                                                                                                           learn_rate,
                                                                                                           weight_decay,
                                                                                                           drop_rate)))
                    plt.clf()
                    vae_model.save_model(os.path.join(os.getcwd(),
                                                      'resnet_vaeV2_stage_1_LD{}_LR{:.0E}_WD{:.0E}_DR{:.1f}.pt'.format(latent_dim,
                                                                                                      learn_rate,
                                                                                                      weight_decay,
                                                                                                      drop_rate)))

do_search_simple_vae = False
if do_search_simple_vae:
    max_acc = 0
    model_list = glob.glob(os.path.join(os.getcwd(), 'ver1/simple_vae*.pt'))
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.2", ]
    torch.manual_seed(0)
    np.random.seed(0)
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
    for i_model in range(len(model_list)):
        data_loader.reset()
        feat, classes, spk_data = vae_model.forward_encoder(data_loader, 1e9)
        vae_model = SimpleVae.load_vae_model(model_list[i_model])

        unique_classes, class_counts = np.unique(classes, return_counts=True)
        small_labels = unique_classes[class_counts < 70]
        spk_data = spk_data[(classes != small_labels).all(axis=1), :, :]
        feat = feat[(classes != small_labels).all(axis=1), :]
        classes = classes[(classes != small_labels).all(axis=1)]

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model],good_clusters=data_loader.good_clusters)
        # print('model {} had acc of {}'.format(model_list[i_model], classifier2.gmm_acc))
        # if classifier2.gmm_acc > max_acc:
        #     max_acc = classifier2.gmm_acc
        #     best_model = i_model

do_search_resnet_2d = True
if do_search_resnet_2d:
    model_list = glob.glob(os.path.join(os.getcwd(), 'ver1/resnet_vae_stage_*.pt'))
    max_acc = 0
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.2", ]
    torch.manual_seed(0)
    np.random.seed(0)
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
    for i_model in range(len(model_list)):
        data_loader.reset()
        vae_model = resnet_2d_vae.load_vae_model(model_list[i_model])
        feat, classes, spk_data = vae_model.forward_encoder(data_loader, 1e9)

        unique_classes, class_counts = np.unique(classes, return_counts=True)
        small_labels = unique_classes[class_counts < 70]
        spk_data = spk_data[(classes != small_labels).all(axis=1), :, :]
        feat = feat[(classes != small_labels).all(axis=1), :]
        classes = classes[(classes != small_labels).all(axis=1)]

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model],good_clusters=data_loader.good_clusters)
        # print('model {} had acc of {}'.format(model_list[i_model], classifier2.gmm_acc))
        # if classifier2.gmm_acc > max_acc:
        #     max_acc = classifier2.gmm_acc
        #     best_model = i_model


do_search_resnet_2d_v2 = True
if do_search_resnet_2d_v2:
    model_list = glob.glob(os.path.join(os.getcwd(), 'ver1/resnet_vaeV2_stage*.pt'))
    max_acc = 0
    file_dirs = ["C:/DL_data"]
    file_clu_names = ["mF105_10.spk.2", ]
    torch.manual_seed(0)
    np.random.seed(0)
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=8192, shuffle=False)
    for i_model in range(len(model_list)):
        data_loader.reset()
        vae_model = resnet_2d_vae_v2.load_vae_model(model_list[i_model])
        feat, classes, spk_data = vae_model.forward_encoder(data_loader, 1e9)

        unique_classes, class_counts = np.unique(classes, return_counts=True)
        small_labels = unique_classes[class_counts < 70]
        spk_data = spk_data[(classes != small_labels).all(axis=1), :, :]
        feat = feat[(classes != small_labels).all(axis=1), :]
        classes = classes[(classes != small_labels).all(axis=1)]

        classifier2 = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model],good_clusters=data_loader.good_clusters)
        # print('model {} had acc of {}'.format(model_list[i_model], classifier2.gmm_acc))
        # if classifier2.gmm_acc > max_acc:
        #     max_acc = classifier2.gmm_acc
        #     best_model = i_model

