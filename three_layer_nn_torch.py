import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ThreeLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.tanh(out)
        out = self.layer2(out)
        out = self.tanh(out)
        out = self.layer3(out)
        out = self.sigmoid(out)
        return out

def main():
    X, y = load_data('BankNote_Authentication.csv')
    
    np.random.seed(42)
    shuffle_index = np.random.permutation(X.shape[0])
    X, y = X[shuffle_index], y[shuffle_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    # parameters for the model same as the from scratch implementation
    input_size = 4
    hidden_size1 = 4
    hidden_size2 = 4
    output_size = 1
    learning_rate = 0.01
    num_epochs = 1000
    
    # model and loss function
    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    costs = []
    
    print("\nTraining three-layer neural network with PyTorch:")
    for epoch in range(num_epochs):
        # forward prop
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # show loss every 100 epochs
        if (epoch+1) % 100 == 0:
            costs.append(loss.item())
            print(f'Epoch {epoch+1}/{num_epochs}, Cost: {loss.item():.4f}')
    
    # getting preds
    with torch.no_grad():
        
        train_pred = model(X_train)
        train_pred_class = (train_pred > 0.5).float()
        train_acc = (train_pred_class == y_train).float().mean().item() * 100
        
        test_pred = model(X_test)
        test_pred_class = (test_pred > 0.5).float()
        test_acc = (test_pred_class == y_test).float().mean().item() * 100
    
    print("\nModel Performance:")
    print(f"Training Accuracy: {train_acc:.4f}%")
    print(f"Test Accuracy: {test_acc:.4f}%")
    print(f"Final loss: {costs[-1]:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(0, num_epochs, 100), costs)
    plt.title('Three-layer NN (PyTorch - tanh)')
    plt.xlabel('Epochs (hundreds)')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.savefig('pytorch_learning_curve.png')
    plt.show()

if __name__ == "__main__":
    main()