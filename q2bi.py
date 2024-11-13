import numpy as np
import time
import matplotlib.pyplot as plt

def gradient_method_backtracking(f, g, x0, s, alpha, beta, epsilon, max_iter=1000):
    """
    Gradient method with backtracking stepsize rule

    INPUT
    =======================================
    f ........ objective function
    g ........ gradient of the objective function
    x0 ....... initial point
    s ........ initial choice of stepsize
    alpha .... tolerance parameter for the stepsize selection
    beta ..... the constant by which the stepsize is multiplied
               at each backtracking step (0 < beta < 1)
    epsilon .. tolerance parameter for stopping rule
    max_iter . maximum number of iterations to prevent infinite loop (default: 1000)

    OUTPUT
    =======================================
    hist ..... history of the points visited during the optimization
    x ........ optimal solution (up to a tolerance)
               of min f(x)
    fun_val .. optimal function value
    """

    # Check inputs
    if beta <= 0 or beta >= 1:
        raise ValueError('beta must be between 0 and 1')

    # Initialization
    x_prev = np.ones(len(x0)) * np.inf
    x = np.array(x0)
    grad = g(x)
    fun_val = f(x)
    iter_count = 0

    # Gradient descent with backtracking line search
    while np.linalg.norm(x - x_prev) / np.linalg.norm(x) >= epsilon and iter_count < max_iter:
        iter_count += 1
        t = s
        # Backtracking line search
        while fun_val - f(x - t * grad) < alpha * t * np.linalg.norm(grad)**2:
            t = beta * t
        # Update x
        x_prev = x
        x = x - t * grad
        fun_val = f(x)
        grad = g(x)

    return x, iter_count

noisy_data = np.genfromtxt("noisy_signal.csv", delimiter=",")
n = len(noisy_data)

# Create an n x n matrix L with 1s on the main diagonal and -1s on the diagonal above the main diagonal
L = np.zeros((n, n))
np.fill_diagonal(L, 1)
L[np.arange(n - 1), np.arange(1, n)] = -1
# Delete the last row of L
L = L[:-1, :]
L = L * (n - 1)

l = 5e-4
gamma = 1e3
w = 75

A_bar = np.hstack((np.eye(n), np.zeros((n, n - 1))))
L_bar = np.hstack((-L, np.eye(n - 1)))
w_bar = np.hstack((np.zeros((n,)), np.ones((n - 1,)))) * w

def objective_func(z):
    @np.vectorize
    def h(t):
        if np.abs(t) <= 1 / gamma:
            return gamma * (t ** 2) / 2
        else:
            return np.abs(t) - (1 / (2 * gamma))

    def f(u):
        return (np.linalg.norm(A_bar @ u - noisy_data) ** 2) / 2 + (l / 2) * (np.linalg.norm(L_bar @ u - z) ** 2) + w_bar.dot(h(u))

    return f

def gradient_func(z):
    @np.vectorize
    def h_prime(u):
        if u < -1 / gamma:
            return -1
        elif u <= 1 / gamma:
            return gamma * u
        return 1

    def f(u):
        return A_bar.T @ (A_bar @ u - noisy_data) + l * L_bar.T @ (L_bar @ u - z) + (w_bar * h_prime(u))

    return f

# Gradient descent hyperparameters
s = 1e6
alpha = 1e-4
beta = 0.5
epsilon = 1e-4

start = time.time()
z = np.ones(n - 1)
x0 = noisy_data
y0 = L @ x0
u_prev = np.ones(2 * n - 1) * np.inf
u = np.hstack((x0, y0))

iterations = []

total_start_time = time.time()

while np.linalg.norm(u[:n] - u_prev[:n]) / np.linalg.norm(u[:n]) >= 1e-5:
    start_time = time.time()
    u_prev = u
    u, inner_iterations = gradient_method_backtracking(objective_func(z), gradient_func(z), np.hstack(u_prev), s, alpha, beta, epsilon, max_iter=500)
    z = z + (L @ u[:n] - u[n:])

    iterations.append((time.time() - start_time, inner_iterations))

print('----- Results -----')
for i, (time_taken, inner_iterations) in enumerate(iterations):
    print(f"Iteration {i + 1}: Time taken: {time_taken:.2f}s, Inner iterations: {inner_iterations}")
print(f"Time taken: {time.time() - total_start_time:.2f}s")

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(np.linspace(0, n, n), noisy_data, s=2, c='r', label='Noisy Signal')
plt.plot(u[:n], label='Denoised Signal')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.title('Noisy vs Denoised Signal')
plt.legend()
plt.grid(True)
plt.show()

