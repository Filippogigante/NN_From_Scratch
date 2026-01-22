import numpy as np
import matplotlib.pyplot as plt

class error_plot:

    def plot(self, epochs, err):
        raise NotImplementedError

    
class default_plot(error_plot):
    '''
    The class implements a method to plot
    the Training errors, for each epoch.
    '''
    def plot(self, err):
        plt.figure('Error plot')
        plt.plot(err, c='red', marker='.', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.show()

class val_err_plot(error_plot):
    '''
    The class implements a method to plot both 
    the Validation and the Training errors, for each epoch.
    '''
    def plot(self, val, err, metric_names):
        '''
        plt.figure('Validation, Error Plot')
        plt.plot(err, c='red', linestyle='-', label='Training')
        plt.plot(val, c='blue', linestyle='--', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.legend()
        plt.show()
        '''
        '''
        The class plots different metrics
        '''
        if len(val) != len(err):
            print("Attention the dimension of the two arrays doesn't match")
        else:
            n = len(val)

        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(15, 5))
        axes_flat = axes.flatten()
        for i in range(n):
            axes_flat[i].plot(err[i], c='red', linestyle='-', label='Training')
            axes_flat[i].plot(val[i], c='blue', linestyle='--', label='Validation')
            axes_flat[i].set_title(f"Metric: {metric_names[i]}")
            axes_flat[i].grid(True)
            axes_flat[i].legend()
            axes_flat[i].set_box_aspect(1)

        plt.tight_layout()
        plt.show()