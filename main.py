import numpy as np
from ANNNI_sample_class import ANNNIsampler
from RBM import RBM

sampler = ANNNIsampler(1,0.5,50,0)
sampler.load_data("ANNNI_ising_configurations.npy")
data_set = sampler.states

network = RBM(50,20)
network.train_model(0.01, 1, data_set, 300)

filename = "RBM_weights.npy"
network.save_model(filename)

#network.load_model(filename)

states_RBM = network.generate_rbm_states()
sampler.set_states_RBM(states_RBM)
sampler.display_correlation()


