import numpy as np
import matplotlib.pyplot as plt

noisy_data = np.genfromtxt("noisy_signal.csv", delimiter=",")
n = len(noisy_data)

# Create an n x n matrix L with 1s on the main diagonal and -1s on the diagonal above the main diagonal
L = np.zeros((n, n))
np.fill_diagonal(L, 1)
L[np.arange(n - 1), np.arange(1, n)] = -1

def rls(w, f):
    return np.linalg.inv(np.identity(n) + w * (np.transpose(L) @ L)).dot(f)

# Define the regularization weights
regularization_weights = [10 ** (-4), 5 * 10 ** (-4), 10 ** (-3)]

# Apply the RLS function with different regularization weights and store the results
denoised_signals = [rls(w, noisy_data) for w in regularization_weights]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(noisy_data, label='Noisy Signal')
for i, w in enumerate(regularization_weights):
    plt.plot(denoised_signals[i], label=f'Denoised Signal (w={w})', linestyle='--')

plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.title('Noisy vs Denoised Signal for Different Regularization Weights')
plt.legend()
plt.grid(True)
plt.show()
