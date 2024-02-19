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
        
class LassoRegression:
    def __init__(self, lr:float = 0.01, max_iter:int = 1000, alpha:float = 0.01, tol:float = 1e-6):
        """Lasso Regression Model using Gradient Descent

        Args:
            lr (float, optional): Parameter for Learning-rate. Defaults to 0.01.
            max_iter (int, optional): Maximum iteration for the model to learn. Defaults to 1000.
            alpha (float, optional): Regularization parameter. Defaults to 0.01.
            tol (float, optional): Tolerance for the model to stop learning. Defaults to 1e-6.
        """
        self.max_iter = max_iter
        self.lr = lr
        self.alpha = alpha
        self.tol = tol
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.weights_history = []
    
    def predict(self, X:np.ndarray):
        # y = wX + b
        return np.dot(X, self.weights) + self.bias
    
    def _cost(self, y, y_pred):
        return np.mean((y - y_pred)**2) + self.alpha * np.linalg.norm(self.weights, 1)
        
    def fit(self, X:np.ndarray, y:np.ndarray):
        """Fit the Lasso Regression model to the training data.

        Args:
            X (np.ndarray): The training data with predictors.
            y (np.ndarray): The target values.
        """
        if X.size == 0 or y.size == 0:
                    raise ValueError("Input arrays cannot be empty")

        obs, vars = X.shape
        self.weights = np.zeros(vars)
        self.bias = 0

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            dw = (1/obs) * (np.dot(X.T, (y_pred - y)) + self.alpha * np.sign(self.weights))
            db = (1/obs) * np.sum(y_pred - y)

            if np.all(abs(self.lr * dw) < self.tol):
                break

            self.weights -= self.lr * dw - self.alpha * self.lr * np.sign(self.weights)
            self.bias -= self.lr * db

            self.cost_history.append(self._cost(y, y_pred))
            self.weights_history.append(self.weights.copy())