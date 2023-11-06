import numpy as np

class LinearRegression:
    def __init__(self, lr:float = 0.01, max_iter:int = 1000):
        """Linear Regression Model using Gradient Descent

        Args:
            lr (float, optional): Parameter for Learning-rate. Defaults to 0.01.
            max_iter (int, optional): Maximum iteration for the model to learn. Defaults to 1000.
        """
        self.max_iter = max_iter
        self.lr = lr
        self.weights = None
        self.bias = None
    
    def predict(self, X:np.ndarray):
        # y = wX + b
        return np.dot(X, self.weights) + self.bias
        
    def fit(self, X:np.ndarray, y:np.ndarray):
        """Fit the Linear Regression model to the training data.

        Args:
            X (np.ndarray): The training data with predictors.
            y (np.ndarray): The target values.
        """
        obs, vars = X.shape

        # Initialize Weights and Bias
        self.weights = np.zeros(vars)
        self.bias = 0
        
        # iterate over max_iter
        for i in range(self.max_iter):
            y_pred = self.predict(X)
            
            # Compute Gradient Descent of Weights and Bias
            dw = (1/obs) * np.dot(X.T, (y_pred - y))
            db = (1/obs) * np.sum(y_pred - y)
            
            # Update Weights and Bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            

class LogisticRegression:
    def __init__(self, lr:float = 0.01, max_iter:int = 1000, threshold:float = 0.5):
        """Logistic Sigmoid Regression Model using Gradient Descent

        Args:
            lr (float, optional): Parameter for Learning-rate. Defaults to 0.01.
            max_iter (int, optional): Maximum iteration for the model to learn. Defaults to 1000.
            threshold (float, optional): Threshold for the model to classify. This should range from 0 to 1. Defaults to 0.5.
        """
        self.max_iter = max_iter
        self.lr = lr
        self.threshold = threshold
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z:np.ndarray):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X:np.ndarray):
        # y = sigmoid(wX + b)
        return (self._sigmoid(np.dot(X, self.weights) + self.bias) > self.threshold).astype(int)
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        """Fit the Logistic Regression model to the training data.

        Args:
            X (np.ndarray): The training data with predictors.
            y (np.ndarray): The target values.
        """
        obs, vars = X.shape
        self.weights = np.zeros(vars)
        self.bias = 0
        
        for i in range(self.max_iter):
            y_pred = self.predict(X)
            
            # Compute Gradient Descent of Weights and Bias
            dw = (1/obs) * np.dot(X.T, (y_pred - y))
            db = (1/obs) * np.sum(y_pred - y)
            
            # Update Weights and Bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        