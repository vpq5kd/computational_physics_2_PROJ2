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

    def data_average_dw(self, data_set):
        de_dw_array = np.zeros((self.N, self.M))
        for sigma in data_set:
            tau = self.sample_hidden(sigma)
            for i in range(self.N):
                for j in range(self.M):
                    de_dw_array[i,j] += sigma[i]*tau[j]
        
        de_dw_array /= len(data_set)
        
        return de_dw_array

    def model_average_dw(self, data_set, k):
        
        de_dw_array = np.zeros((self.N, self.M))
        for sigma0 in data_set:
            sigma = sigma0.copy()
            tau = self.sample_hidden(sigma)

            for _ in range(k):
                sigma = self.sample_visible(tau)
                tau = self.sample_hidden(sigma)

            for i in range(self.N):
                for j in range(self.M):
                    de_dw_array[i,j] += sigma[i]*tau[j]

        de_dw_array /= len(data_set)
        
        return de_dw_array

    def update_w(self, eta, k, data_set): 
        self.W += eta * (self.data_average_dw(data_set)-self.model_average_dw(data_set,k))
    def calculate_epsilon_W(self, w_before):
        epsilon_w = 0
        for i in range(self.N):
            for j in range(self.M):
                epsilon_w += np.abs(self.W-w_before)
        return epsilon_w

    def train_model(self, eta, k, data_set, n_epochs):
        for epoch in n_epochs:
            w_before = self.W
            self.update_w(eta,k,data_set)
            epsilon_w = self.calculate_espilon_W(w_before)
            print(epsilon_w)
                       



        
                      
            
    


