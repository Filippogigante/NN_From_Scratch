import numpy as np
import pickle
import os
from Loss import *
from Update import *
from Layer import *
from Error_plots import *
from Regularize import *



class Model:
    def __init__(self, eta, alpha, lamb, layers, update, loss, metric, regularizer):
        self.lamb = lamb
        self.layers = layers
        self.eta = eta
        self.alpha = alpha

        update_map = {
            "standard": lambda: std_update(self.eta),
            "momentum": lambda: momentum_update(self.eta, self.alpha)
        }

        loss_metric_map = {
            "mse": mse,
            "mee": mee,
            "binary_cross_entropy" : binary_cross_entropy,
            "accuracy" : accuracy,
            "error" : error
        }

        regularizer_map = {
            "l1": lambda: l1(self.lamb),
            "l2": lambda: l2(self.lamb)
        }

        if isinstance(update, str):
            try:
                self.update = update_map[update.lower()]()
                print(f"Update algorithm set to: {self.update}")
            except KeyError:
                raise ValueError(f"Unknown Gradient update '{update}'. Available: {list(update_map.keys())}")
        else:
            raise BaseException("Update must be a string")

        if isinstance(loss, str):
            try:
                self.loss = loss_metric_map[loss.lower()]()
                print(f"Loss set to: {self.loss}")
            except KeyError:
                raise ValueError(f"Unknown Loss '{loss}'. Available: {list(loss_metric_map.keys())}")
        else:
            raise BaseException("loss must be a string")

        if isinstance(metric, str):
            try:
                self.metric = loss_metric_map[metric.lower()]()
                print(f"Metric set to: {self.metric}")
            except KeyError:
                raise ValueError(f"Unknown Metric '{metric}'. Available: {list(loss_metric_map.keys())}")
        else:
            raise BaseException("metric must be a string")
        
        if isinstance(regularizer, str):
            try:
                self.regularizer = regularizer_map[regularizer.lower()]()
                print(f"Regularize algorithm set to: {self.regularizer}")
            except KeyError:
                raise ValueError(f"Unknown Regularizer '{regularizer}'. Available: {list(regularizer_map.keys())}")
        else:
            raise BaseException("Regularizer must be a string")

    def add_layer(self, layer):
        return self.layers.append(layer)

    def forward_pass_model(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
    
    def train(self, x, y, batch_size):
        layer_inputs = []
        layer_outputs = []
        # Forward prop
        for layer in self.layers:
            layer_inputs.append(x)
            x = layer.forward_pass(x)
            layer_outputs.append(x)

        # Backprop
        delta = self.loss.backward_loss(layer_outputs[-1], y)
        for i, (inp, layer) in enumerate(zip(reversed(layer_inputs), reversed(self.layers))):
            skip = ((i == 0) and isinstance(layer, sigmoid))
            delta = layer.backward_pass(delta, inp, batch_size, skip)
            
        # Aggiornamento Pesi
        for layer in self.layers:
            if (layer.get_type() != "dropout"):
                self.update.update(layer, self.regularizer)
            

    def fit(self, epochs, x, y, x_val, y_val, batch_size, plot = False):

        full_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"

        n_train = x.shape[1]
        n_val = x_val.shape[1]
        X_loc = x.copy()
        Y_loc = y.copy()
        err = []
        val = [1000]
        best_val_loss = float('inf')
        Flag = False
        threshold = 0.8

        for epoch in range(epochs+1):
            
            #Train the model using all batches
            for k in range(0, n_train, batch_size):
                X_batch = X_loc[:, k : k + batch_size]
                Y_batch = Y_loc[:, k : k + batch_size]
                
                self.train(X_batch, Y_batch, X_batch.shape[1])
            
            #if ((epoch % 5) == 0):
             #   indices = np.arange(x.shape[1])
              #  np.random.shuffle(indices)
               # X_loc = X_loc[:, indices]
                #Y_loc = Y_loc[:, indices]

            #Val loss is the total validation loss for each batch
            val_loss = 0

            for l in range(0, n_val, batch_size):

                X_batch = x_val[:, l : l + batch_size]
                Y_batch = y_val[:, l : l + batch_size]
                val_loss += self.evaluate(X_batch, Y_batch)

            #We compute the average val loss of the various batches
            if batch_size < n_val:
                avg_val_loss = val_loss / (n_val / batch_size)
            else:
                avg_val_loss = val_loss
                
                
            if ((epoch % 500) == 0):
                #self.eta *= 0.95
                #self.update.set_eta(self.eta)
                o = self.forward_pass_model(x)
                e = self.metric.forward_loss(o, y)
                print(f"Epoch: {epoch}/{epochs}")
                print("Errore: ", e)
            
            if avg_val_loss - val[-1] > threshold:
                Flag = True
                print(f'----GRAFICO VENUTO MALE (Validation Error irregolare)----')
                break
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                data_to_save = {
                "weights": self.all_layers_weights(),
                "bias": self.all_layers_bias()
                }
                with open(full_path, 'wb') as file:  
                    pickle.dump(data_to_save, file)

            o = self.forward_pass_model(x)
            e = self.metric.forward_loss(o, y)
            err.append(e)
            val.append(avg_val_loss)
        
        val.pop(0)
        if not Flag:
            print(f"-------Best Val for this Configuration: {best_val_loss}")

        if plot:
            plot = val_err_plot()
            plot.plot(val, err)
            
        return best_val_loss, Flag
    
    
        
    def evaluate(self, x, y):
        """
        Computes the error of the model, given the input data x
        """
        o = self.forward_pass_model(x)
        e = self.metric.forward_loss(o, y)
        return e
    
    def all_layers_weights(self):
        W = []
        for layer in self.layers:
            if layer.get_type() != "dropout":
                W.append(layer.get_weights())
        return W

    def all_layers_bias(self):
        b = []
        for layer in self.layers:
            if layer.get_type() != "dropout":
                b.append(layer.get_bias())
        return b

    def summary(self):
        neurons = 0
        n_params = 0
        for layer in self.layers:
            neurons += layer.dim_input
            n_params += layer.get_weights().size + layer.get_bias().size
        
        print(f"--------------------------------------------------------------------")
        print(f"Number of neurons: {neurons};", f"Number of parameters: {n_params}.")
        print(f"Number of Mbyte: {(n_params * 8) / 1024}")
        print(f"Learning rate: {self.eta}", f"Momentum coefficient: {self.alpha}")
        print(f"--------------------------------------------------------------------")
