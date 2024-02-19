import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, problem:str = 'classification'):
        """Feed forward Neural Network with two hidden layers.
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
        self.l1 = nn.Linear(input_size, self.hidden_size) # input layer
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size) # hidden layer
        self.l3 = nn.Linear(self.hidden_size, self.output_size) # hidden layer
        self.relu = nn.ReLU() # activation function
        self.problem = problem

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
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
        
class CNN(nn.Module):
    def __init__(self, output_size, device:str = 'cpu'):
        """Convolutional Neural Network with two hidden layers.
        This CNN class can be used for classification.
        It uses PyTorch's nn.Module class as a base class.

        Args:
            output_size (int): The total number of neurons at the output layer.
            This should be equal to the number of pixels of flatten image.
        """
        super(CNN, self).__init__() # super() function makes class inheritance more manageable and extensible
        self.device = device
        self.output_size = output_size
        self.conv1 = nn.Conv2d(3, 32, 3, device=self.device) # input layer
        self.conv2 = nn.Conv2d(32, 32, 3, device=self.device) # hidden layer
        self.conv3 = nn.Conv2d(32, 64, 3, device=self.device) # hidden layer
        self.conv4 = nn.Conv2d(64, 64, 3, device=self.device) # hidden layer
        self.fc1 = nn.Linear(1600, 128, device=self.device) # hidden layer
        self.fc2 = nn.Linear(128, self.output_size, device=self.device) # hidden layer
        self.relu = nn.ReLU() # activation function
        self.maxpool = nn.MaxPool2d(2, 2) # pooling layer

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        if self.output_size == 1:
            out = torch.sigmoid(out) # activation function
            
        elif self.output_size > 1:
            out = torch.softmax(out, dim=1)

        return out
    
    def predict(self, X):
        return self.forward(X)
    
    def fit(self, train_loader, max_iter:int = 1000, lr:float = 0.01, momentum:float = 0.9, weight_decay:float = 0.005):
        """Fit the CNN model to the training data. The training data should be in the form of a DataLoader object.
        Args:
            train_loader (torchvision.datasets.folder.ImageFolder): Pytorch DataLoader object.
            max_iter (int, optional): Maximum iterations or epochs to train. Defaults to 1000.
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum to be passed to torch.optim.SGD. Defaults to 0.9.
            weight_decay (float, optional): Weight decay to be passed to torch.optim.SGD. Defaults to 0.005.
        """
        self.max_iter = max_iter
        self.lr = lr
        self.loss = []
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=momentum, weight_decay=weight_decay)
        
        for epoch in range(self.max_iter):
            for i, (images, labels) in enumerate(train_loader):  
                
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.forward(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            self.loss.append(loss.item())
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.max_iter, loss.item()))
            
class UNet(nn.Module):
    def __init__(self, device:str = 'cpu'):
        """UNet Model for biomedical image segmentation using Adam's optimizer.
        This Unet class can be used for biomedical image segmentation.
        It uses PyTorch's nn.Module class as a base class.

        Args:
            input_size (int): The total number of neurons at the input layer.
            This should be equal the number of pixels after flattened.
        """
        super(UNet, self).__init__() # super() function makes class inheritance more manageable and extensible
        self.device = device
        self.conv1 = nn.Conv2d(1, 64, 3, device=self.device)
        self.conv2 = nn.Conv2d(64, 64, 3, device=self.device)
        self.conv3 = nn.Conv2d(64, 128, 3, device=self.device)
        self.conv4 = nn.Conv2d(128, 128, 3, device=self.device)
        self.conv5 = nn.Conv2d(128, 256, 3, device=self.device)
        self.conv6 = nn.Conv2d(256, 256, 3, device=self.device)
        self.conv7 = nn.Conv2d(256, 512, 3, device=self.device)
        self.conv8 = nn.Conv2d(512, 512, 3, device=self.device)
        self.conv9 = nn.Conv2d(512, 1024, 3, device=self.device)
        self.conv10 = nn.Conv2d(1024, 1024, 3, device=self.device)
        self.conv11 = nn.Conv2d(1024, 512, 3, device=self.device)
        self.conv12 = nn.Conv2d(512, 512, 3, device=self.device)
        self.conv13 = nn.Conv2d(512, 256, 3, device=self.device)
        self.conv14 = nn.Conv2d(256, 256, 3, device=self.device)
        self.conv15 = nn.Conv2d(256, 128, 3, device=self.device)
        self.conv16 = nn.Conv2d(128, 128, 3, device=self.device)
        self.conv17 = nn.Conv2d(128, 64, 3, device=self.device)
        self.conv18 = nn.Conv2d(64, 64, 3, device=self.device)
        self.conv19 = nn.Conv2d(64, 1, 1, device=self.device)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.maxpool(out)
        out = nn.Dropout2d(p=0.25)(out)
        
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.maxpool(out)
        out = nn.Dropout2d(p=0.5)(out)
        
        out = self.relu(self.conv5(out))
        out = self.relu(self.conv6(out))
        out = self.maxpool(out)
        out = nn.Dropout2d(p=0.5)(out)
        
        out = self.relu(self.conv7(out))
        out = self.relu(self.conv8(out))
        out = self.maxpool(out)
        out = nn.Dropout2d(p=0.5)(out)
        
        out = self.relu(self.conv9(out))
        out = self.relu(self.conv10(out))
        
        out = self.upsample(out)
        out = nn.Dropout2d(p=0.5)(out)
        out = self.relu(self.conv11(out))
        out = self.relu(self.conv12(out))
        
        out = self.upsample(out)
        out = nn.Dropout2d(p=0.5)(out)
        out = self.relu(self.conv13(out))
        out = self.relu(self.conv14(out))
        
        out = self.upsample(out)
        out = nn.Dropout2d(p=0.5)(out)
        out = self.relu(self.conv15(out))
        out = self.relu(self.conv16(out))
        
        out = self.upsample(out)
        out = nn.Dropout2d(p=0.5)(out)
        out = self.relu(self.conv17(out))
        out = self.relu(self.conv18(out))
        out = self.conv19(out)
        
        return out
    
    def predict(self, X):
        return self.forward(X)
    
    def fit(self, X, y, max_iter:int = 1000, lr:float = 0.01, weight_decay:float = 0.005):
        """Fit the UNet model to the training data.

        Args:
            X (torch.tensor): The training data with predictors.
            y (torch.tensor): The target values.
            max_iter (int, optional): The total number of iteration for the model to train. Defaults to 1000.
            lr (float, optional): The model's learning-rate. Defaults to 0.01.
            weight_decay (float, optional): Weight decay to be passed to torch.optim.SGD. Defaults to 0.005.
        """
        self.max_iter = max_iter
        self.lr = lr
        self.loss = []
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        
        if isinstance(X, torch.Tensor):
            X = X.to(self.device)
        else:
            X = torch.tensor(X).to(self.device)
            
        if isinstance(y, torch.Tensor):
            y = y.to(self.device)
        else:
            y = torch.tensor(y).to(self.device)
        
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
                
        