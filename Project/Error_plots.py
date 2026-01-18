import numpy as np
import matplotlib.pyplot as plt

class error_plot:

    def plot(self, epochs, err):
        raise NotImplementedError

    
class default_plot(error_plot):
    
    def plot(self, err):
        plt.figure('Error plot')
        plt.plot(err, c='red', marker='.', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.show()

class val_err_plot(error_plot):

    def plot(self, val, err):
        plt.figure('Validation, Error Plot')
        plt.plot(err, c='red', linestyle='-', label='Training')
        plt.plot(val, c='blue', linestyle='-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.legend()
        plt.show()