import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def energy(state, J1, J2):
    n = len(state)
    E = 0
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

    return 2 * s * (
        -J1 * (state[im1] + state[ip1])
        +J2 * (state[im2] + state[ip2])
    )

def metropolis_zero_T(n, J1, J2, max_steps=200000):
    state = rng.choice([-1, 1], size=n)

    for _ in range(max_steps):
        i = rng.integers(n)
        dE = delta_energy_flip(state, i, J1, J2)

        if dE < 0:
            state[i] *= -1
        elif dE == 0:
            state[i] *= -1

    return state

def correlation_function(state):
    n = len(state)
    max_x = n // 2
    C = np.zeros(max_x)

    for x in range(max_x):
        C[x] = np.mean(state * np.roll(state, -x))

    return C

def main():
    N = 100
    J1 = 1.0
    J2 = 0.5
    num_configs = 10

    corr_arr = np.zeros(N // 2)
    energies = []

    for config in range(num_configs):
        state = metropolis_zero_T(N, J1, J2)
        corr_arr += correlation_function(state)
        energies.append(energy(state, J1, J2))
        print(config)

    corr_arr /= num_configs

    print("Unique final energies:", np.unique(energies))
    print("C(0) =", corr_arr[0])

    x_arr = np.arange(N // 2)

    plt.figure()
    plt.plot(x_arr, corr_arr, marker='o', ms=4)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$C(x)=\langle \sigma_i \sigma_{i+x}\rangle$")
    plt.show()

main()
