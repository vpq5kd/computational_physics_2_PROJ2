import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

def energy(state, J1, J2):
    n = len(state)
    E = 0.0
    for i in range(n):
        E += -J1 * state[i] * state[(i + 1) % n]
        E +=  J2 * state[i] * state[(i + 2) % n]
    return E

def delta_energy_flip(state, i, J1, J2):
    n = len(state)
    s = state[i]

    im1 = (i - 1) % n
    ip1 = (i + 1) % n
    im2 = (i - 2) % n
    ip2 = (i + 2) % n
    dE = 2*s * (
            J1 * (state[im1] + state[ip1])
        -J2 * (state[im2] + state[ip2])
    )
    return dE

def transition_probability(delta_energy_flip,T):
    if delta_energy_flip <= 0:
        return 1.0 
    if T==0:
        return 0.0
    return np.exp(-delta_energy_flip / T)

def metropolis(n, j1, j2, t):
    ising_start_state = np.random.choice([-1,1], size=n)
    state = ising_start_state
    for i in range(0, 500000):
        index_to_flip = rng.integers(n)
        
        de = delta_energy_flip(state, index_to_flip, j1, j2)  
        probability = transition_probability(de, t) 
        
        accept = rng.random() < probability
        if accept: 
            state[index_to_flip] *= -1
       
    print(energy(state, j1, j2))
    return state

def main():
    N = 100
    j1 = 1 
    j2 = 0.5
    t = 0
    max_x = 35

    num_configs = 50
    corr_arr = np.zeros(max_x)

    for config in range(num_configs):
        state = metropolis(N, j1, j2, t)
        print(state)
        for x in range(max_x):
            corr = 0
            for i in range(N):
                j = (i+x) % N
                corr += state[i] * state[j]
            corr_arr[x] += corr / N
        print(config)
 
    corr_arr /= num_configs

    x_arr = np.arange(max_x)
    
    plt.figure()
    plt.plot(x_arr, corr_arr, color='magenta', marker='o')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$C(x) = \langle \sigma_i \sigma_{i+x} \rangle$")
    plt.show()   

main()



        
