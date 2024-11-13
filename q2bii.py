import numpy as np
import time
import matplotlib.pyplot as plt

l = 1e-2
w = 2e-3

noisy_data = np.genfromtxt("noisy_signal.csv", delimiter=",")
n = len(noisy_data)

# Create an n x n matrix L with 1s on the main diagonal and -1s on the diagonal above the main diagonal
L = np.zeros((n, n))
np.fill_diagonal(L, 1)
L[np.arange(n - 1), np.arange(1, n)] = -1
# Delete the last row of L
L = L[:-1, :]
L = L * (n - 1)

x = noisy_data
y = L @ x
z = np.ones(n - 1)

def get_next_x(x, y, z):
    return np.linalg.inv(np.identity(n) + l * (L.T @ L)) @ (noisy_data + l * (L.T @ (y - z)))

def get_next_y(x, y, z):
    def s(t, n):
        return t / abs(t) * max(abs(t) - n, 0)

    new_y = np.zeros(n - 1)
    for i, row in enumerate(L):
        new_y[i] = s(row @ x + z[i], w / l)

    return new_y

def get_next_z(x, y, z):
    return z + (L @ x - y)

start_time = time.time()

iter_count = 0
while True:
    next_x = get_next_x(x, y, z)
    next_y = get_next_y(next_x, y, z)
    next_z = get_next_z(next_x, next_y, z)

    iter_count += 1

    if np.linalg.norm(next_x - x) / np.linalg.norm(next_x) < 1e-5:
        break

    x, y, z = next_x, next_y, next_z

print(f'Number of iterations" {iter_count}')
print(f'Time taken: {time.time() - start_time} seconds')
# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(np.linspace(0, n, n), noisy_data, s=2, c='r', label='Noisy Signal')
plt.plot(x, label='Denoised Signal')

plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.title('Noisy vs Denoised Signal')
plt.legend()
plt.grid(True)
plt.show()

