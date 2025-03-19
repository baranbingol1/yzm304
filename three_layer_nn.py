from utils import sigmoid
import numpy as np

def init_params(n_x, n_h1, n_h2, n_y=1):
    """Initialize parameters for a three-layer neural network using He initialization."""
    rng = np.random.default_rng(seed=42)

    # he initialization for better gradient flow in deeper networks instead of *0.01
    W1 = rng.standard_normal((n_h1, n_x)) * np.sqrt(2./n_x)
    b1 = np.zeros((n_h1, 1))
    W2 = rng.standard_normal((n_h2, n_h1)) * np.sqrt(2./n_h1)
    b2 = np.zeros((n_h2, 1))
    W3 = rng.standard_normal((n_y, n_h2)) * np.sqrt(2./n_h2)
    b3 = np.zeros((n_y, 1))

    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }

    return params

def forward_prop(X, params, activation_func):
    """Perform forward propagation."""
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    Z1 = W1 @ X + b1
    A1 = activation_func(Z1)
    Z2 = W2 @ A1 + b2
    A2 = activation_func(Z2)
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2,
        'Z3': Z3,
        'A3': A3
    }

    return A3, cache

def compute_cost(A3, y):
    """Computes the cross-entropy cost with numerical stability."""
    m = y.shape[1]
    # add small epsilon to avoid log(0)
    epsilon = 1e-8
    cost = -np.sum(y * np.log(A3 + epsilon) + (1 - y) * np.log(1 - A3 + epsilon)) / m
    return cost

def back_prop(X, y, cache, params, activation_func_derivative=None):
    """Perform back propagation."""
    m = X.shape[1]  # number of samples
    W2, W3 = params['W2'], params['W3']
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    
    # output layer gradients
    dZ3 = A3 - y  # Shape: (1, m)
    dW3 = np.dot(dZ3, A2.T) / m  # Shape: (1, n_h2)
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m  # Shape: (1, 1)
    
    # second hidden layer gradients
    dZ2 = np.dot(W3.T, dZ3) * activation_func_derivative(cache['Z2'])
    
    dW2 = np.dot(dZ2, A1.T) / m  # Shape: (n_h2, n_h1)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # Shape: (n_h2, 1)
    
    # first hidden layer gradients
    dZ1 = np.dot(W2.T, dZ2) * activation_func_derivative(cache['Z1'])
        
    dW1 = np.dot(dZ1, X.T) / m  # Shape: (n_h1, n_x)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # Shape: (n_h1, 1)
    
    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
        'dW3': dW3,
        'db3': db3
    }
    
    return grads

def update_params(params, grads, learning_rate=0.01):
    """Update parameters using gradient descent."""
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    b1, b2, b3 = params['b1'], params['b2'], params['b3']
    dW1, dW2, dW3 = grads['dW1'], grads['dW2'], grads['dW3']
    db1, db2, db3 = grads['db1'], grads['db2'], grads['db3']

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    new_params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }

    return new_params

def train(X, y, n_h1, n_h2, activation_func, activation_func_derivative=None, num_iterations=1000, learning_rate=0.01, print_cost=True, print_interval=100):
    """Trains a three-layer neural network."""
    n_x = X.shape[0]
    n_y = y.shape[0]
    costs = []
    
    params = init_params(n_x, n_h1, n_h2, n_y)
    
    # GD
    for i in range(num_iterations):
        A3, cache = forward_prop(X, params, activation_func)
        cost = compute_cost(A3, y)
        grads = back_prop(X, y, cache, params, activation_func_derivative)
        params = update_params(params, grads, learning_rate)
        
        if i % print_interval == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")
    
    return params, costs

def predict(X, params, activation_func, threshold=0.5):
    """Uses a trained neural network to make predictions."""
    
    A3, _ = forward_prop(X, params, activation_func)
    
    # convert probs to binary predictions
    predictions = (A3 >= threshold).astype(int)
    
    return predictions

def evaluate(predictions, y):
    """Evaluates the performance of the model."""
    m = y.shape[1]
    accuracy = np.sum(predictions == y) / m * 100
    return accuracy