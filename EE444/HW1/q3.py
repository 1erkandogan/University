import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def erlang_b(c, rho):
    return (rho**c / np.prod(range(1, c+1))) / np.sum([rho**k / np.prod(range(1, k+1)) for k in range(c+1)])


def buffer_size(P):
    rho_vals = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7,
                0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
    buffer_sizes = []
    for rho in rho_vals:
        c = 1
        while erlang_b(c, rho) > P:
            c += 1
        buffer_sizes.append(c * 1000)
    return rho_vals, buffer_sizes


rho_vals, buffer_sizes_1 = buffer_size(1e-6)
rho_vals, buffer_sizes_2 = buffer_size(1e-12)

plt.plot(rho_vals, buffer_sizes_1, label='P = 10^-6')
plt.plot(rho_vals, buffer_sizes_2, label='P = 10^-12')
plt.xlabel('ρ')
plt.ylabel('Buffer Size (Bytes)')
plt.legend()
plt.show()
