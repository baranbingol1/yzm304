from two_layer_nn import train as two_layer_nn_train, predict as two_layer_predict, evaluate as two_layer_evaluate
from three_layer_nn import train as three_layer_nn_train, predict as three_layer_predict, evaluate as three_layer_evaluate
from utils import load_data, sigmoid, tanh
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def tanh_derivative(x):
    """Derivative of the tanh function."""
    return 1 - np.power(tanh(x), 2)

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

# load the data
X, y = load_data('BankNote_Authentication.csv')

# shuffle the data
np.random.seed(42)
shuffle_index = np.random.permutation(X.shape[0])
X, y = X[shuffle_index], y[shuffle_index]

# split the data by train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train = X_train.T
y_train = y_train.reshape(1, -1)
X_test = X_test.T
y_test = y_test.reshape(1, -1)

# initialize results dictionary
results = {
    "Model": [],
    "Training Cost": [],
    "Training Accuracy (%)": [],
    "Test Accuracy (%)": []
}

# train and evaluate two-layer neural network with tanh
print("\nTraining two-layer neural network with tanh activation:")
two_layers_params_tanh, two_layers_costs_tanh = two_layer_nn_train(
    X_train, y_train, n_h=4, activation_func=tanh, activation_func_derivative=tanh_derivative,
    num_iterations=1000, learning_rate=0.01, print_cost=True, print_interval=100
)

train_predictions_2l_tanh = two_layer_predict(X_train, two_layers_params_tanh, tanh)
test_predictions_2l_tanh = two_layer_predict(X_test, two_layers_params_tanh, tanh)
train_accuracy_2l_tanh = two_layer_evaluate(train_predictions_2l_tanh, y_train)
test_accuracy_2l_tanh = two_layer_evaluate(test_predictions_2l_tanh, y_test)

results["Model"].append("Two-layer NN (tanh)")
results["Training Cost"].append(two_layers_costs_tanh[-1])
results["Training Accuracy (%)"].append(train_accuracy_2l_tanh)
results["Test Accuracy (%)"].append(test_accuracy_2l_tanh)

# train and evaluate two-layer neural network with sigmoid
print("\nTraining two-layer neural network with sigmoid activation:")
two_layers_params_sigmoid, two_layers_costs_sigmoid = two_layer_nn_train(
    X_train, y_train, n_h=4, activation_func=sigmoid, activation_func_derivative=sigmoid_derivative,
    num_iterations=1000, learning_rate=0.01, print_cost=True, print_interval=100
)

train_predictions_2l_sigmoid = two_layer_predict(X_train, two_layers_params_sigmoid, sigmoid)
test_predictions_2l_sigmoid = two_layer_predict(X_test, two_layers_params_sigmoid, sigmoid)
train_accuracy_2l_sigmoid = two_layer_evaluate(train_predictions_2l_sigmoid, y_train)
test_accuracy_2l_sigmoid = two_layer_evaluate(test_predictions_2l_sigmoid, y_test)

results["Model"].append("Two-layer NN (sigmoid)")
results["Training Cost"].append(two_layers_costs_sigmoid[-1])
results["Training Accuracy (%)"].append(train_accuracy_2l_sigmoid)
results["Test Accuracy (%)"].append(test_accuracy_2l_sigmoid)

# train and evaluate three-layer neural network with tanh
print("\nTraining three-layer neural network with tanh activation:")
three_layers_params_tanh, three_layers_costs_tanh = three_layer_nn_train(
    X_train, y_train, n_h1=4, n_h2=4, activation_func=tanh, 
    activation_func_derivative=tanh_derivative, num_iterations=1000, 
    learning_rate=0.01, print_cost=True, print_interval=100
)

train_predictions_3l_tanh = three_layer_predict(X_train, three_layers_params_tanh, tanh)
test_predictions_3l_tanh = three_layer_predict(X_test, three_layers_params_tanh, tanh)
train_accuracy_3l_tanh = three_layer_evaluate(train_predictions_3l_tanh, y_train)
test_accuracy_3l_tanh = three_layer_evaluate(test_predictions_3l_tanh, y_test)

results["Model"].append("Three-layer NN (tanh)")
results["Training Cost"].append(three_layers_costs_tanh[-1])
results["Training Accuracy (%)"].append(train_accuracy_3l_tanh)
results["Test Accuracy (%)"].append(test_accuracy_3l_tanh)

# train and evaluate three-layer neural network with sigmoid
print("\nTraining three-layer neural network with sigmoid activation:")
three_layers_params_sigmoid, three_layers_costs_sigmoid = three_layer_nn_train(
    X_train, y_train, n_h1=4, n_h2=4, activation_func=sigmoid, 
    activation_func_derivative=sigmoid_derivative, num_iterations=1000, 
    learning_rate=0.01, print_cost=True, print_interval=100
)

train_predictions_3l_sigmoid = three_layer_predict(X_train, three_layers_params_sigmoid, sigmoid)
test_predictions_3l_sigmoid = three_layer_predict(X_test, three_layers_params_sigmoid, sigmoid)
train_accuracy_3l_sigmoid = three_layer_evaluate(train_predictions_3l_sigmoid, y_train)
test_accuracy_3l_sigmoid = three_layer_evaluate(test_predictions_3l_sigmoid, y_test)

results["Model"].append("Three-layer NN (sigmoid)")
results["Training Cost"].append(three_layers_costs_sigmoid[-1])
results["Training Accuracy (%)"].append(train_accuracy_3l_sigmoid)
results["Test Accuracy (%)"].append(test_accuracy_3l_sigmoid)

# create results dataframe and display as table
results_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.4f'))

# plot learning curves
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, 1000, 100), two_layers_costs_tanh)
plt.title('Two-layer NN (tanh)')
plt.xlabel('Iterations (hundreds)')
plt.ylabel('Cost')

plt.subplot(2, 2, 2)
plt.plot(np.arange(0, 1000, 100), two_layers_costs_sigmoid)
plt.title('Two-layer NN (sigmoid)')
plt.xlabel('Iterations (hundreds)')
plt.ylabel('Cost')

plt.subplot(2, 2, 3)
plt.plot(np.arange(0, 1000, 100), three_layers_costs_tanh)
plt.title('Three-layer NN (tanh)')
plt.xlabel('Iterations (hundreds)')
plt.ylabel('Cost')

plt.subplot(2, 2, 4)
plt.plot(np.arange(0, 1000, 100), three_layers_costs_sigmoid)
plt.title('Three-layer NN (sigmoid)')
plt.xlabel('Iterations (hundreds)')
plt.ylabel('Cost')

plt.tight_layout()
plt.savefig('learning_curves.png')
plt.show()