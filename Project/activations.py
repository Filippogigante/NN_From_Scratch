import numpy as np

class ActivationFunctions():
    
    """
    Abstract class for activation functions
    """

    def forward(self, x):
        raise NotImplementedError 
        
    def backward(self, x):
        raise NotImplementedError


class relu(ActivationFunctions):

    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return ((x > 0).astype(float))


class tanh(ActivationFunctions):

    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - self.forward(x) ** 2.0


class sigmoid(ActivationFunctions):

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))
    

class softmax(ActivationFunctions):

    def forward(self, x):
        return np.exp(x) / (np.sum(np.exp(x), axis=0, keepdims=True))
    
    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))
 
    
class identity(ActivationFunctions):

    def forward(self, x):
        return x
    
    def backward(self, x):
        return 1.0
