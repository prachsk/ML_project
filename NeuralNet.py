import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, problem:str = 'classification'):
        """Feed forward Neural Network with a single hidden layer.
        This ANN class can be used for both classification and regression problems.
        It uses PyTorch's nn.Module class as a base class.

        Args:
            input_size (int): The total number of neurons at the input layer.
            This should be equal the number of features in used as predictors.
            hidden_size (int): The total number of neurons at the hidden layer.
            output_size (int): The total number of neurons at the output layer.
            This should be equal to the number of classes for multi-classes classification problems and 1 for regression and binary classification problems.
            problem (str, optional): The type of problem that the ANN is designe to do. Can be either 'classification' or 'regression'.
            To train ANN for a binary classification problem, set "output_size" to 1.
            Defaults to 'classification'
        """
        super(ANN, self).__init__() # super() function makes class inheritance more manageable and extensible
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_size, hidden_size) # input layer
        self.l2 = nn.Linear(hidden_size, self.output_size) # hidden layer
        self.relu = nn.ReLU() # activation function
        self.problem = problem

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        
        if self.problem == 'classification' and self.output_size == 1:
            out = torch.sigmoid(out) # activation function
            
        elif self.problem == 'classification' and self.output_size > 1:
            out = torch.softmax(out)

        return out
    
    def predict(self, X):
        return self.forward(X)
    
    def fit(self, X, y, max_iter:int = 1000, lr:float = 0.01):
        """Fit the ANN model to the training data.

        Args:
            X (torch.tensor): The training data with predictors.
            y (torch.tensor): The target values.
            max_iter (int, optional): The total number of iteration for the model to train. Defaults to 1000.
            lr (float, optional): The model's learning-rate. Defaults to 0.01.
        """
        self.max_iter = max_iter
        self.lr = lr
        self.loss = []
        
        if self.problem == 'classification':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        
        for epoch in range(self.max_iter):
            # Forward pass
            outputs = self.forward(X)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss.append(loss.item())
            if (epoch+1) % 100 == 0:
                print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.max_iter, loss.item()))
        
        