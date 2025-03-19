from utils import sigmoid
import numpy as np

def init_params(n_x, n_h, n_y=1):
    """Initialize parameters for a two-layer neural network with He initialization."""
    rng = np.random.default_rng(seed=42)

    # weights and biases for first layer [shape: W1(n_h, n_x), b1(n_h, 1)]
    W1 = rng.standard_normal((n_h, n_x)) * np.sqrt(2./n_x)
    b1 = np.zeros((n_h, 1))
    
    # weights and biases for output layer [shape: W2(n_y, n_h), b2(n_y, 1)]
    W2 = rng.standard_normal((n_y, n_h)) * np.sqrt(2./n_h)
    b2 = np.zeros((n_y, 1))

    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return params

def forward_prop(X, params, activation_func):
    """Perform forward propagation.
    
    Args:
        X: input features [shape: (n_x, m)]
        params: model parameters
        activation_func: activation function for hidden layer
        
    Returns:
        A2: output predictions [shape: (n_y, m)]
        cache: cached values for backpropagation
    """
    W1, b1 = params['W1'], params['b1']  # shapes: W1(n_h, n_x), b1(n_h, 1)
    W2, b2 = params['W2'], params['b2']  # shapes: W2(n_y, n_h), b2(n_y, 1)

    # first layer forward pass
    Z1 = W1 @ X + b1  # shape: (n_h, m)
    A1 = activation_func(Z1)  # shape: (n_h, m)
    
    # output layer forward pass
    Z2 = W2 @ A1 + b2  # shape: (n_y, m)
    A2 = sigmoid(Z2)  # shape: (n_y, m)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache

def compute_cost(A2, y):
    """Computes cross-entropy cost with numerical stability.
    
    Args:
        A2: predictions [shape: (n_y, m)]
        y: ground truth labels [shape: (n_y, m)]
        
    Returns:
        cost: scalar cross-entropy loss
    """
    m = y.shape[1]  # number of examples
    epsilon = 1e-8  # small value to avoid log(0)
    cost = -np.sum(y * np.log(A2 + epsilon) + (1 - y) * np.log(1 - A2 + epsilon)) / m
    return cost

def back_prop(X, y, cache, params, activation_func_derivative=None):
    """Perform back propagation.
    
    Args:
        X: input features [shape: (n_x, m)]
        y: ground truth labels [shape: (n_y, m)]
        cache: cached values from forward pass
        params: model parameters
        activation_func_derivative: derivative of hidden layer activation (default: tanh derivative)
        
    Returns:
        grads: gradients for all parameters
    """
    m = X.shape[1]  # number of examples
    W1, W2 = params['W1'], params['W2']  # shapes: W1(n_h, n_x), W2(n_y, n_h)
    A1, A2 = cache['A1'], cache['A2']  # shapes: A1(n_h, m), A2(n_y, m)
    
    # output layer gradients
    dZ2 = A2 - y  # shape: (n_y, m)
    dW2 = np.dot(dZ2, A1.T) / m  # shape: (n_y, n_h)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # shape: (n_y, 1)
    
    # hidden layer gradients - use provided derivative or default to tanh derivative
    if activation_func_derivative is not None:
        dZ1 = np.dot(W2.T, dZ2) * activation_func_derivative(cache['Z1'])  # shape: (n_h, m)
    else:
        # default to tanh derivative: 1 - tanh²(x) = 1 - A1²
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # shape: (n_h, m)
        
    dW1 = np.dot(dZ1, X.T) / m  # shape: (n_h, n_x)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # shape: (n_h, 1)
    
    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    
    return grads

def update_params(params, grads, learning_rate=0.01):
    """Update parameters using gradient descent.
    
    Args:
        params: current parameters
        grads: computed gradients
        learning_rate: learning rate for gradient descent
        
    Returns:
        new_params: updated parameters
    """
    W1, W2 = params['W1'], params['W2']  # shapes: W1(n_h, n_x), W2(n_y, n_h)
    b1, b2 = params['b1'], params['b2']  # shapes: b1(n_h, 1), b2(n_y, 1)
    dW1, dW2 = grads['dW1'], grads['dW2']  # shapes: dW1(n_h, n_x), dW2(n_y, n_h)
    db1, db2 = grads['db1'], grads['db2']  # shapes: db1(n_h, 1), db2(n_y, 1)

    # update rule: param = param - learning_rate * gradient
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    new_params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return new_params

def train(X, y, n_h, activation_func, activation_func_derivative=None, num_iterations=1000, learning_rate=0.01, print_cost=True, print_interval=100):
    """Trains a two-layer neural network.
    
    Args:
        X: input features [shape: (n_x, m)]
        y: ground truth labels [shape: (n_y, m)]
        n_h: number of hidden units
        activation_func: activation function for hidden layer
        activation_func_derivative: derivative of activation function (optional)
        num_iterations: number of training iterations
        learning_rate: learning rate for gradient descent
        print_cost: whether to print cost during training
        print_interval: interval for printing cost
        
    Returns:
        params: trained parameters
        costs: list of costs during training
    """
    n_x = X.shape[0]  # number of features
    n_y = y.shape[0]  # number of outputs
    costs = []
    
    # initialize parameters with He initialization
    params = init_params(n_x, n_h, n_y)
    
    # gradient descent
    for i in range(num_iterations):
        # forward propagation
        A2, cache = forward_prop(X, params, activation_func)
        
        # compute cost
        cost = compute_cost(A2, y)
        
        # backward propagation
        grads = back_prop(X, y, cache, params, activation_func_derivative)
        
        # update parameters
        params = update_params(params, grads, learning_rate)
        
        # print cost
        if i % print_interval == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")
    
    return params, costs

def predict(X, params, activation_func, threshold=0.5):
    """Uses a trained neural network to make predictions.
    
    Args:
        X: input features [shape: (n_x, m)]
        params: trained parameters
        activation_func: activation function for hidden layer
        threshold: threshold for binary classification
        
    Returns:
        predictions: binary predictions [shape: (n_y, m)]
    """
    # forward propagation
    A2, _ = forward_prop(X, params, activation_func)
    
    # convert probabilities to binary predictions
    predictions = (A2 >= threshold).astype(int)
    
    return predictions

def evaluate(predictions, y):
    """Evaluates the performance of the model.
    
    Args:
        predictions: model predictions [shape: (n_y, m)]
        y: ground truth labels [shape: (n_y, m)]
        
    Returns:
        accuracy: classification accuracy (percentage)
    """
    m = y.shape[1]  # number of examples
    accuracy = np.sum(predictions == y) / m * 100
    return accuracy