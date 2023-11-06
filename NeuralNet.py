import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, problem:str = 'classification'):
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
    
    def fit(self, X, y, max_iter:int = 1000, lr:float = 0.01, optimizer:str = 'SGD'):
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
        
        