import numpy as np

class regularizer:
    '''
    Abstract class to add the Tichonov regularization to the model
    '''
    def __init__(self, lamb):
        self.lamb = lamb
        pass

    def fp(self, W):
        return NotImplementedError   

    def bp(self, W):
        return NotImplementedError

   
class l1(regularizer):
    def __init__(self, lamb):
        super().__init__(lamb)

    def fp(self, W):
        return self.lamb * np.sum(np.abs(W), axis=1, keepdims=True)
    
    def bp(self, W):
        return self.lamb * np.sign(W)

    
class l2(regularizer):
    def __init__(self, lamb):
        super().__init__(lamb)

    def fp(self, W):
        return 0.5 * self.lamb * np.sum(np.power(W, 2.0), axis=1, keepdims=True)
    
    def bp(self, W):
        return self.lamb * W
