import matplotlib.pyplot as plt 
import numpy as np
from ANNNI_sample_class import ANNNIsampler

sampler = ANNNIsampler(1,.5,100,0)
sampler.generate_data()

sampler.display_correlation("ANNNI_only.png")
