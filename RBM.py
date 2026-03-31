import numpy as np
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.rng = np.random.default_rng()
        
        self.W = 0.01 * self.rng.standard_normal(N,M)
        self.theta_v = np.zeros(N)
        self.theta_h = np.zeros(M)

    def E_RBM(self, sigma, tau):
        energy = 0
        
        for i in range(self.N):
            for j in range(self.M):
                  energy -= self.W[i,j]*sigma[i]*tau[j]

        for i in range(self.N):
            energy += self.theta_v[i]*sigma[i]
        
        for j in range(self.M):
            energy += self.theta_h[j]*tau[j]          

        return energy

    def sample_hidden(self, sigma):
        tau = np.zeros(self.M)

        for j in range(self.M):
            h_j = 0.0
            for i in range(self.N):
                h_j += self.W[i,j] *sigma[i]
            h_j -= self.theta_h[j]

            prob = 1.0/(1.0 +np.exp(-2*h_j))
            
            if self.rng.random() < prob:
                tau[j] = 1
            else:
                tau[j] = -1
        
        return tau
    
    def sample_visible(self, tau):
        sigma = np.zeros(self.N)
    
        for i in range(self.N):
            h_i = 0.0
            for j in range(self.M):
                h_i += self.W[i,j] * tau[j]
            h_i -= self.theta_v[i]

            prob = 1/0/(1.0 *np.exp(-2*h_i))

            if self.rng.random() < prob:
                sigma[i] = 1
            else:
                sigma[i] = -1

        return sigma


            
    


