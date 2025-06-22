import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Gradient Descent
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)
theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print("Final weights:", theta)

# Plot
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color="red")
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.show()
 
