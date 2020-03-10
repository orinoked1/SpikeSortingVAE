# SpikeSortingVAE

We developed an innovative spike sorting feature extractor variational autoencoder model. The encoder network is equipped with residual transformations to extract representative features from spikes. The latent space is used as a feature space for a Gaussian Mixture Model that separates the spikes to different clusters. The clustering accuracy and performance of the proposed feature space is compared to the traditionally used Principal Component space. 

Experimental results on in-vivo dataset show that the proposed approach consistently outperforms the conventional Principal Component approach. With the latent space compared to the Principal Component space, the Gaussian Mixture Model accuracy tested against ground truth was higher by 15%. Two common clustering quality indices, the Davies-Boulding index and the Calinski-Harabasz index, also showed that clustering at the latent space was superior to the clustering at the Principal Component space.

Code:

The main function is main.py it does training, validation, results evaluation and creates figures.

For data handling it use the SpikeDataLoader class defined in dataHandle.py. We uses the “spk” and “clu” file format (http://neurosuite.sourceforge.net/formats.html) for data storing. An example data files can be found in “example_data” folder (small data file of 10k spikes). The SpikeDataLoader class can be replaced by any iterable object that returns spikes and classes as torch tensors.

The model itself is defined as a pytorch model in autoEncoder.py. Previously saved models (such as ours in model sub-directory) can be loaded using the load_vae_model class method. Model parameters are accessible using the cfg dictionary in the main function.

ClassificationTester.py performs all performance evaluation.
