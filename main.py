import numpy as np
from ANNNI_sample_class import ANNNIsampler
from RBM import RBM

sampler = ANNNIsampler(1,0.5,50,0)
sampler.load_data("ANNNI_ising_configurations.npy")
data_set = sampler.states

network = RBM(50,20)
network.train_model(0.01, 1, data_set, 100)


