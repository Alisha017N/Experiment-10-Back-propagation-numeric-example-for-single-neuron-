
import numpy as np

# Single sample input and target output
x = np.array([0.5, -1.2])
y = 1.0

# Initial weights, bias, and learning rate
w = np.array([0.1, -0.2])
b = 0.0
lr = 0.1

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass
z = np.dot(w, x) + b
a = sigmoid(z)

# Calculate loss
loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))

# Backward pass: calculate gradients
dz = a - y
dw = dz * x
db = dz

# Update weights and bias
w_new = w - lr * dw
b_new = b - lr * db

# Display results
print("a (output):", round(a, 4), "loss:", round(loss, 4))
print("dw:", np.round(dw, 4), "db:", round(db, 4))
print("Updated w:", np.round(w_new, 4), "Updated b:", round(b_new, 4))
