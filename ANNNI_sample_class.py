import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def _sample_one(args):
    j1, j2, N, T, steps = args
    sampler = ANNNIsampler(j1,j2,N,T)
    return  sampler.sample(steps=steps)

class ANNNIsampler:
    
    rng = np.random.default_rng()
    
    states = np.array([])
    states_RBM = np.array([])

    def __init__(self, j1, j2, N, T):
        self.j1 = j1
        self.j2 = j2
        self.N = N
        self.T = T
    
    def set_states_RBM(self, states_RBM):
        self.states_RBM = states_RBM

    def delta_energy_flip(self, state, i):
        n = len(state)
        s = state[i]

        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        im2 = (i - 2) % n
        ip2 = (i + 2) % n
        dE = 2*s * (
                self.j1 * (state[im1] + state[ip1])
            -self.j2 * (state[im2] + state[ip2])
        )
        return dE
    
    def transition_probability(self, de):
        if de <= 0:
            return 1.0
        if self.T==0:
            return 0.0
        return np.exp(-de / self.T)
    
    def sample(self, steps=500000):
        ising_start_state = np.random.choice([-1,1], size=self.N)
        state = ising_start_state
        for i in range(0, steps):
            index_to_flip = self.rng.integers(self.N)

            de = self.delta_energy_flip(state, index_to_flip)
            probability = self.transition_probability(de)

            accept = self.rng.random() < probability
            if accept:
                state[index_to_flip] *= -1

        return state

    def generate_data(self,num_samples=1000, steps=500000):
        args = [(self.j1, self.j2, self.N, self.T, steps)]*num_samples
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            states = list(tqdm(executor.map(_sample_one, args), total=num_samples, desc="Generating samples..."))
        self.states = np.array(states)
        return self.states
    def save_data(self, filename):
        if self.states.size > 0:
            np.save(filename, self.states)
            print(f"Saved states in {filename}.")
        else:
            print(f"Please generate or load states before saving.")

    def load_data(self, filename):
        try:
            self.states = np.load(filename)
        except OSError:
            print(f"{filename} is empty or corrupted, please try again.")
    
    def calculate_correlation(self, states):
        max_x = 35
        corr_arr = np.zeros(max_x)
        for state in states:
            for x in range(max_x):
                corr = 0
                for i in range(self.N):
                    j = (i+x) % self.N
                    corr += state[i] * state[j]
                corr_arr[x] += corr / self.N
        corr_arr /= len(states)
        return corr_arr, max_x

    def display_correlation(self):
        plt.figure()
        if self.states.size > 0: 
            annni_states_corr_arr, max_x = self.calculate_correlation(self.states)
            
            x_arr = np.arange(max_x)

            plt.plot(x_arr, annni_states_corr_arr, color='magenta', marker='o', label='ANNNI')
        elif self.states.size == 0: 
            print(f"{self.states} must have data to be displayed!")

        if self.states_RBM.size >0:
            rbm_states_corr_arr, max_x = self.calculate_correlation(self.states_RBM)
            x_arr = np.arange(max_x)
            
            plt.plot(x_arr, rbm_states_corr_arr, color='forestgreen', marker='o', label='RBM')

        elif self.states_RBM.size == 0: 
            print(f"{self.states_RBM} must have data to be displayed!")
        
        
        plt.xlabel(r"$x$")
        plt.ylabel(r"$C(x) = \langle \sigma_i \sigma_{i+x} \rangle$")
        plt.legend()
        plt.show()
'''
def main():
    AS = ANNNIsampler(1,1/2,50,0)
    filename = "ANNNI_ising_configurations.npy"
    AS.load_data(filename)
    AS.display_correlation()
main()
'''
