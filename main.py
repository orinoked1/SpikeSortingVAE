from dataHandle import SpikeDataLoader
from autoEncoder import Vae
from ClassificationTester import ClassificationTester
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy

# load data and train on a grid using different hyper params
do_train = True
if do_train:
    batch_size = 2048
    shuffle = True
    file_dirs = [os.path.join(os.getcwd(), 'example_data')]
    file_clu_names = ["mF105_10.spk.2"]
    data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=batch_size, shuffle=shuffle)

    cfg = {"n_channels": data_loader.N_CHANNELS_OUT, "spk_length": data_loader.N_SAMPLES,
           "conv1_ch": 256, "conv2_ch": 16, "latent_dim": 36,
           "conv0_ker": 1, "conv1_ker": 3, "conv2_ker": 1,
           "ds_ratio_1": 2, "ds_ratio_2": 2,
           "cardinality": 32, "dropRate": 0.0, "n_epochs": 15,
           "learn_rate": 1e-3, "weight_decay": 1e-5,
           "shuffle_channels": False}
    cfg["ds_ratio_tot"] = cfg["ds_ratio_2"] * cfg["ds_ratio_1"]
    torch.manual_seed(0)
    np.random.seed(0)
    # training
    vae_model = Vae(cfg)
    unique_labels, target_means = vae_model.calc_means(data_loader)
    target_means = np.random.randn(target_means.shape[0], target_means.shape[1])
    target_means = target_means / np.linalg.norm(target_means, axis=1)[:, None]
    factors = [25, 50, 75]
    LRs = [3e-4, 3e-5]
    conv1_chs = [64, 128]
    # train
    for conv1_ch in conv1_chs:
        for fact in factors:
            for lr in LRs:
                cfg["conv1_ch"] = conv1_ch
                vae_model = Vae(cfg)

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
                                         'vaeStage3_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}_F{}_FLC{}.png'.format(
                                             cfg["latent_dim"],
                                             lr,
                                             cfg["weight_decay"],
                                             cfg["shuffle_channels"],
                                             cfg["dropRate"],
                                             fact,
                                             cfg["conv1_ch"])))
                plt.clf()
                vae_model.save_model(os.path.join(os.getcwd(),
                                                  'vaeStage3_LD{}_LR{:.0E}_WD{:.0E}_SD{}_DR{:.1f}_F{}_FLC{}.pt'.format(
                                                      cfg["latent_dim"],
                                                      lr,
                                                      cfg["weight_decay"],
                                                      cfg["shuffle_channels"],
                                                      cfg["dropRate"],
                                                      fact,
                                                      cfg["conv1_ch"])))
                del vae_model
# loop over trained moder and write performance to a *.csv file
do_search = True
if do_search:
    max_acc = 0
    model_list = glob.glob(os.path.join('model', 'Model.pt'))
    for i_model in range(len(model_list)):
        file_dirs = [os.path.join(os.getcwd(), 'example_data')]
        file_clu_names = ["mF105_10.spk.2", ]
        torch.manual_seed(0)
        np.random.seed(0)
        data_loader = SpikeDataLoader(file_dirs, file_clu_names, batch_size=2048, shuffle=False)
        vae_model = Vae.load_vae_model(model_list[i_model])
        feat, classes, all_spk_idx = vae_model.forward_encoder(data_loader, 1e9)
        unique_classes, class_counts = np.unique(classes, return_counts=True)
        classifier = ClassificationTester(feat, classes, use_pca=False, name=model_list[i_model],good_clusters=data_loader.good_clusters)
        print('model {} had acc of {}'.format(model_list[i_model], classifier.gmm_acc))
        if classifier.gmm_acc > max_acc:
            max_acc = classifier.gmm_acc
            best_model = i_model

# write a performance  a *.csv file for the GT feature space (spored in an *.npy file)
classify_gt=True
if classify_gt:
    # GT features (3PC + energy per channel) are saved in an npy file (supplied a-priori for testing)
    fet_1 = np.load(os.path.join(os.getcwd(), 'example_data','mF105_10.fet.2.npy'))
    clu_data = scipy.io.loadmat(os.path.join(os.getcwd(), 'example_data','mF105_10.clu.2.mat'))
    clu = clu_data['clu']
    clu = clu[1:]
    full_map_table = np.load(os.path.join(os.getcwd(), 'example_data','mapmF105_10.npy'))
    good_clusters = full_map_table[full_map_table[:, 1] == 2, 2]
    classifier = ClassificationTester(fet_1, clu, use_pca=False, name="PCA",good_clusters=good_clusters)
    print('model {} had acc of {}'.format("PCA", classifier.gmm_acc))




