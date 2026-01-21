import numpy as np
import math

class loss:

    def forward_loss(self, x, y):
        raise NotImplementedError
    
    def backward_loss(self, x, y):
        raise NotImplementedError
        
class mse(loss):

    def forward_loss(self, output, target):
        return np.mean(np.sum(np.power(output - target, 2), axis=0, keepdims=True))
    
    def backward_loss(self, output, target):
        return output - target
    
class mee(loss):

    def forward_loss(self, output, target):
        return np.mean(np.sqrt(np.sum(np.power(output - target, 2), axis=0, keepdims=True)))
    
    def backward_loss(self, output, target):
        return (output - target) / self.forward_loss(output, target)
    
    
class binary_cross_entropy(loss):
    
    
    def forward_loss(self, output, target, margin):
        """
        Computes BCE loss
        
        this forward loss is not used in the training loop (only needs the backward),
        it's used to compute the error of our model and plotted in the graph
        """
        return np.mean(-np.sum(target * np.log(output) + (1 - target) * np.log(1 - output), axis=1))
        
        
    def backward_loss(self, output, target):
        """
        Computes backward loss of the BCE function, 
        in our case this is directly ∂L/∂z of the last layer, so we skip the last layer backward pass (∂o/∂z)
        
        y_hat - y is the result of the reduction of ∂L/∂o * ∂L/∂z
        
        :param output: the output of our last layer 
        :param y: target in the dataset
        """
        return output - target
        
class accuracy(loss):
    def forward_loss(self,  y_pred,y_true, threshold=0.5):
        """
        Computes binary accuracy
        y_true: shape (1, n_samples)
        y_pred: shape (1, n_samples)
        """
    
        # Convert probabilities to 0/1
        y_pred_labels = (y_pred >= threshold).astype(int)
        
        # Ensure shapes match
        if y_pred_labels.shape != y_true.shape:
            y_pred_labels = y_pred_labels.reshape(y_true.shape)

        # Count correct predictions
        correct = (y_pred_labels == y_true).sum()
        


        # Divide by number of samples to get fraction
        accuracy = correct / y_true.shape[1]  # y_true.shape[1] = number of samples
        return accuracy


class error(loss):
    def forward_loss(self,  y_pred,y_true, threshold=0.5):
        """
        Computes binary accuracy
        y_true: shape (1, n_samples)
        y_pred: shape (1, n_samples)
        """
       
        y_pred_labels = (y_pred >= threshold).astype(int)
        
        if y_pred_labels.shape != y_true.shape:
            y_pred_labels = y_pred_labels.reshape(y_true.shape)
       
        correct = (y_pred_labels == y_true).sum()

        accuracy = correct / y_true.shape[1]  
        return 1-accuracy
