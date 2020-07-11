import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from collections import Counter
import os.path


class ClassificationTester(object):
    def __init__(self, spikes, labels, use_pca, n_init=5, max_iter=100, model_name = ""):
        self.n_init = n_init
        self.max_iter = max_iter
        self.model_name = model_name
        if len(spikes.shape) == 3:
            n_samples = spikes.shape[0]
            n_channels = spikes.shape[1]
            self.features = np.zeros((n_samples, n_channels * 3))
            spikes_energy = np.mean(spikes ** 2, axis=2)
            spikes_energy = spikes_energy[:, :, None]
            spikes /= spikes_energy
            for i_channel in range(n_channels):
                pca = PCA(n_components=3)
                self.features[:, i_channel * 3:(i_channel + 1) * 3] = pca.fit_transform(
                    spikes[:, i_channel, :].squeeze())
        else:
            self.features = spikes
        # self.features = StandardScaler().fit_transform(self.features)
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]
        self.use_pca = use_pca
        self.labels = labels.squeeze()
        self.unique_classes, self.class_counts = np.unique(self.labels, return_counts=True)
        self.n_classes = len(self.unique_classes)
        self.mu, self.sigma, self.inv_sigma = self.calc_mu_sigma()
        self.ID = self.calc_ID()
        self.gmm, self.gmm_classes = self.fit_gmm()
        self.gmm_classes = fit_labels(np.copy(self.labels), self.gmm_classes)
        self.gmm_acc = accuracy_score(self.labels, self.gmm_classes)
        self.DBS = davies_bouldin_score(self.features, self.labels)  # the lower the better
        self.CHS = calinski_harabasz_score(self.features, self.labels)  # the higher the better?
        self.n_classes_for_gmm, self.gmm_acc_with_n_classes = self.n_classes_to_fit_gmm()
        # self.SS = silhouette_score(self.features, self.labels)  # the higher the better
        self.gmm_pairwise_acc = self.fit_gmm_pairwise()
        a = 1

    def calc_mu_sigma(self):
        mu = np.zeros((self.n_features, self.n_classes))
        sigma = np.zeros((self.n_features, self.n_features, self.n_classes))
        inv_sigma = np.zeros((self.n_features, self.n_features, self.n_classes))
        for i_class in range(len(self.unique_classes)):
            label = self.unique_classes[i_class]
            mu[:, i_class] = np.mean(self.features[self.labels == label, :], axis=0)
            sigma[:, :, i_class] = np.cov(self.features[self.labels == label, :].T)
            inv_sigma[:, :, i_class] = np.linalg.inv(sigma[:, :, i_class].squeeze())
        return mu, sigma, inv_sigma

    def calc_ID(self):
        ID = np.zeros(self.n_classes)
        for i_class in range(len(self.unique_classes)):
            label = self.unique_classes[i_class]
            noise_spikes = self.features[self.labels != label, :]
            mu = self.mu[:, i_class].T
            mu = mu[None, :]
            mahal_dist = spatial.distance.cdist(mu, noise_spikes, metric='mahalanobis',
                                                VI=self.inv_sigma[:, :, i_class].squeeze()).squeeze() ** 2
            mahal_dist = np.sort(mahal_dist)
            ID[i_class] = mahal_dist[self.class_counts[i_class]]
        return ID

    def fit_gmm(self):
        gmm = GaussianMixture(n_components=self.n_classes, n_init=self.n_init, max_iter=self.max_iter)
        gmm.fit(self.features)
        gmm_classes = gmm.predict(self.features)
        return gmm, gmm_classes

    def fit_gmm_pairwise(self):
        n_classes = self.n_classes
        pairwise_acc = np.zeros((n_classes, n_classes))
        for i_class_1 in range(n_classes):
            label_1 = self.unique_classes[i_class_1]
            for i_class_2 in range(n_classes):
                if i_class_2 > i_class_1:
                    label_2 = self.unique_classes[i_class_2]
                    features = self.features[(self.labels == label_1) | (self.labels == label_2), :]
                    labels = self.labels[(self.labels == label_1) | (self.labels == label_2)]
                    gmm = GaussianMixture(n_components=2, n_init=self.n_init, max_iter=self.max_iter)
                    gmm.fit(features)
                    gmm_classes = gmm.predict(features)
                    gmm_classes = fit_labels(labels, gmm_classes)
                    pairwise_acc[i_class_1, i_class_2] = accuracy_score(labels, gmm_classes)
        return pairwise_acc

    def n_classes_to_fit_gmm(self):
        n_classes = self.n_classes
        th = 0.8
        N = n_classes * 10
        step = 5
        prev_acc = []
        model_per_csv_name = 'Model performance.csv'
        if not os.path.isfile(model_per_csv_name):
            cols = ['Model', 'Clusters', 'accuracy', 'per_uniform_class', 'mean_uni_mat', 'spikes_in_uni_classes', 'per spikes in uni classes']
            pd.DataFrame(columns=cols).to_csv(model_per_csv_name)


        for i_n_class in range(n_classes, N, step):
            spikes_in_uni_classes = 0
            spike_th = ((self.features.shape[0]) / i_n_class) * 0.05
            gmm = GaussianMixture(n_components=i_n_class, n_init=self.n_init, max_iter=self.max_iter)
            gmm.fit(self.features)
            gmm_classes = gmm.predict(self.features)
            gt_labels = np.copy(self.labels)
            gt_unique_classes = np.unique(gt_labels)
            gt_labels, gmm_classes = self.filter_small_clusters_spikes(spike_th, gt_labels, gmm_classes)
            tested_unique_classes, tested_classes_count = np.unique(gmm_classes, return_counts=True)
            idx = np.argsort(-tested_classes_count)
            tested_unique_classes = tested_unique_classes[idx]
            con_mat = contingency_matrix(gt_labels, gmm_classes)
            uni_mat = np.zeros((i_n_class,1))
            tested_labels_remap = np.copy(gmm_classes)
            for i_class in range(len(tested_unique_classes)):
                new_label_idx = np.argmax(con_mat[:, [tested_unique_classes[i_class]]])
                tested_labels_remap[gmm_classes == tested_unique_classes[i_class]] = gt_unique_classes[new_label_idx]
                uni_mat[tested_unique_classes[i_class]] = con_mat[new_label_idx, [tested_unique_classes[i_class]]] / sum(con_mat[:, [tested_unique_classes[i_class]]])
                if uni_mat[tested_unique_classes[i_class]] > th:
                    spikes_in_uni_classes += con_mat[new_label_idx, [tested_unique_classes[i_class]]]
                con_mat[:, tested_unique_classes[i_class]] = -1
                # con_mat[new_label_idx, :] = -1
            per_uniform_class = sum(uni_mat > th) / i_n_class
            mean_uni_mat = np.mean(uni_mat)
            acc = accuracy_score(gt_labels, tested_labels_remap)
            print(i_n_class, acc, per_uniform_class, mean_uni_mat, spikes_in_uni_classes, spikes_in_uni_classes /self.features.shape[0])
            new_line = {'Model': self.model_name.split("\\")[-1].split('.')[0],
                        'Clusters': [i_n_class],
                        'accuracy': [acc],
                        'per_uniform_class': [per_uniform_class],
                        'mean_uni_mat': [mean_uni_mat],
                        'spikes_in_uni_classes': [spikes_in_uni_classes],
                        'per spikes in uni classes': [spikes_in_uni_classes /self.features.shape[0]]
                        }
            pd.DataFrame(new_line).to_csv(model_per_csv_name, mode='a', header=False)
            prev_acc.append(acc)
            if len(prev_acc) > 3:
                if np.mean([x-y for x, y in zip(prev_acc[-3:], prev_acc[-4:])]) < 0.005:
                    return i_n_class - step * 3

        return i_n_class, acc

    def filter_small_clusters_spikes(self, th, y_true, y_pred):
        c = Counter(y_pred)
        filter_idx = []
        filter_idx += [np.where(c == i) for i in c if c[i] < th]
        filter_idx = np.squeeze(filter_idx)
        y_pred = np.delete(y_pred, filter_idx)
        y_true = np.delete(y_true, filter_idx)
        if len(filter_idx):
            print(len(filter_idx), 'Spikes Deleted')
        return y_true, y_pred

    def plot_2d_pca(self, class2display=-1):
        if class2display == -1:
            class_list = range(self.n_classes)
        else:
            class_list = range(class2display)
        if self.features.shape[1] != 2:
            if self.use_pca:
                pcs = PCA(n_components=2).fit_transform(self.features)
                all_labels = self.labels
            else:
                for i_class in class_list:
                    curr_label = self.unique_classes[i_class]
                    curr_features = self.features[self.labels == curr_label, :]
                    curr_labels = self.labels[self.labels == curr_label]
                    curr_features = curr_features[np.random.permutation(curr_features.shape[0]), :]
                    curr_features = curr_features[:min(curr_features.shape[0], 1000), :]
                    curr_labels = curr_labels[:min(curr_labels.shape[0], 1000)]
                    if i_class == 0:
                        all_features = curr_features
                        all_labels = curr_labels
                    else:
                        all_features = np.concatenate((all_features, curr_features), axis=0)
                        all_labels = np.concatenate((all_labels, curr_labels), axis=0)

                pcs = TSNE(n_components=2).fit_transform(all_features)
        else:
            pcs = self.features
            all_labels = self.labels
        plt.figure()

        for i_class in class_list:
            label = self.unique_classes[i_class]
            pc1 = pcs[all_labels == label, 0]
            pc2 = pcs[all_labels == label, 1]
            plt.scatter(pc1[:min(len(pc1), 100)], pc2[:min(len(pc2), 100)])
        plt.title('{:.2f}'.format(self.gmm_acc))
        plt.show()

    def plot_2d_pca_mat(self, other_gmm_pairwise_acc, class2display=-1):

        if class2display == -1:
            n_classes = self.n_classes
        else:
            n_classes = class2display
        fig, axs = plt.subplots(n_classes, n_classes)
        for i_class_1 in range(n_classes):
            label_1 = self.unique_classes[i_class_1]
            for i_class_2 in range(n_classes):
                if i_class_2 >= i_class_1:
                    label_2 = self.unique_classes[i_class_2]
                    if self.use_pca:  # in pca use all the data
                        dim_reduction = PCA(n_components=2)
                        features = self.features[(self.labels == label_1) | (self.labels == label_2), :]
                        labels = self.labels[(self.labels == label_1) | (self.labels == label_2)]
                    else:  # in TSNE use max 1k for eac class
                        dim_reduction = TSNE(n_components=2)
                        features_1 = self.features[(self.labels == label_1), :]
                        labels_1 = self.labels[(self.labels == label_1)]
                        features_1 = features_1[np.random.permutation(features_1.shape[0]), :]
                        features_1 = features_1[:min(features_1.shape[0], 1000), :]
                        labels_1 = labels_1[:min(labels_1.shape[0], 1000)]
                        features_2 = self.features[(self.labels == label_2), :]
                        labels_2 = self.labels[(self.labels == label_2)]
                        features_2 = features_2[np.random.permutation(features_2.shape[0]), :]
                        features_2 = features_2[:min(features_2.shape[0], 1000), :]
                        labels_2 = labels_2[:min(labels_2.shape[0], 1000)]
                        features = np.concatenate((features_1, features_2), axis=0)
                        labels = np.concatenate((labels_1, labels_2), axis=0)

                    pcs_all = dim_reduction.fit_transform(features)

                    pcs = pcs_all[labels == label_1, :]
                    pcs = pcs[np.random.permutation(pcs.shape[0]), :]
                    axs[i_class_1, i_class_2].scatter(pcs[:min(pcs.shape[0], 100), 0], pcs[:min(pcs.shape[0], 100), 1],
                                                      s=6)
                    axs[i_class_1, i_class_2].tick_params(axis='x', which='both', bottom=False, top=False,
                                                          labelbottom=False)
                    axs[i_class_1, i_class_2].tick_params(axis='y', which='both', left=False, right=False,
                                                          labelleft=False)
                    if i_class_2 != i_class_1:
                        pcs = pcs_all[labels == label_2, :]
                        pcs = pcs[np.random.permutation(pcs.shape[0]), :]
                        axs[i_class_1, i_class_2].scatter(pcs[:min(pcs.shape[0], 100), 0],
                                                          pcs[:min(pcs.shape[0], 100), 1], s=6)
                        if self.gmm_pairwise_acc[i_class_1, i_class_2] > other_gmm_pairwise_acc[
                            i_class_1, i_class_2] + 0.025:
                            title_color = 'green'
                        elif self.gmm_pairwise_acc[i_class_1, i_class_2] < other_gmm_pairwise_acc[
                            i_class_1, i_class_2] - 0.025:
                            title_color = 'red'
                        else:
                            title_color = 'black'
                        axs[i_class_1, i_class_2].set_title(
                            '{:.2f}'.format(self.gmm_pairwise_acc[i_class_1, i_class_2]), fontsize=15,
                            color=title_color)
                else:
                    axs[i_class_1, i_class_2].axis('off')
        plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, hspace=0.3)
        plt.show()
        a = 1


def fit_labels(gt_labels, tested_labels):
    gt_unique_classes = np.unique(gt_labels)
    tested_unique_classes, tested_classes_count = np.unique(tested_labels, return_counts=True)
    idx = np.argsort(-tested_classes_count)
    tested_unique_classes = tested_unique_classes[idx]
    con_mat = contingency_matrix(gt_labels, tested_labels)
    tested_labels_remap = np.copy(tested_labels)
    for i_class in range(len(tested_unique_classes)):
        new_label_idx = np.argmax(con_mat[:, [tested_unique_classes[i_class]]])
        tested_labels_remap[tested_labels == tested_unique_classes[i_class]] = gt_unique_classes[new_label_idx]
        con_mat[:, tested_unique_classes[i_class]] = -1
        con_mat[new_label_idx, :] = -1
    return tested_labels_remap
