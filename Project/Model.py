import numpy as np
from Loss import *
from Update import *
from Layer import *
from Error_plots import *
from Regularize import *
from utilities import save_model



class Model:
    def __init__(self, eta, alpha, lamb, layers, update, loss, validation_loss, metrics, regularizer):
        self.lamb = lamb
        self.layers = layers
        self.eta = eta
        self.alpha = alpha
        self.metrics = []
        self.metric_names = metrics
        self.loss_name = loss

        update_map = {
            "standard": lambda: std_update(self.eta),
            "momentum": lambda: momentum_update(self.eta, self.alpha)
        }

        loss_metric_map = {
            "mse": mse,
            "mee": mee,
            "binary_cross_entropy" : binary_cross_entropy,
            "binary_accuracy" : binary_accuracy,
            "binary_error" : binary_error
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
        
        if isinstance(validation_loss, str):
            try:
                self.validation_loss = loss_metric_map[validation_loss.lower()]()
                print(f"Validation Loss set to: {self.validation_loss}")
            except KeyError:
                raise ValueError(f"Unknown Validation Loss '{validation_loss}'. Available: {list(loss_metric_map.keys())}")
        else:
            raise BaseException("validation loss must be a string")

        for metric in metrics:
            if isinstance(metric, str):
                try:
                    self.metrics.append(loss_metric_map[metric.lower()]())
                    print(f"Metric set to: {metric}")
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
        '''
        Computes the forward for all the layers of the model
        '''
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
    
    def train(self, x, y, batch_size):
        '''
        This methods computes the backward propagation for all the layers and updates the weights.
        (It trains the model on the specific batch)
        '''
        layer_inputs = []
        layer_outputs = []

        # Forward prop
        for layer in self.layers:
            layer_inputs.append(x)
            x = layer.forward_pass(x)
            layer_outputs.append(x)

        # Back prop
        delta = self.loss.backward_loss(layer_outputs[-1], y)
        for i, (inp, layer) in enumerate(zip(reversed(layer_inputs), reversed(self.layers))):
            skip = ((i == 0) and isinstance(layer, sigmoid))
            delta = layer.backward_pass(delta, inp, batch_size, skip)
            
        # Update the weights
        for layer in self.layers:
            if (layer.get_type() != "dropout"):
                self.update.update(layer, self.regularizer)
            

    def fit(self, epochs, x, y, x_val, y_val, batch_size, eta_descent=False, plot=False):
        '''
        This method trains the model on all the data, and computes the validation error, 
        and automatically saves the weights of the best model on the validation error.
        '''

        path_model_best_weights = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"
        patience = 5
        patience_count = 0
        n_train = x.shape[1]
        n_val = x_val.shape[1]
        X_loc = x.copy()
        Y_loc = y.copy()

        # We build the arrays that will contain the validation and error for the plot of the different metrics
        err_metric = []
        val_metric = []
        for metric in self.metrics:
            val_metric.append([])
            err_metric.append([])
        
        val = [1000] # val keeps track of the validation error across epochs (used to check if the validation loss spikes)
        model_best_val_loss = float('inf')
        threshold_flag = False
        threshold = 0.8
        eta_descent_freq = epochs // 10

        for epoch in range(epochs+1):
            
            #Train the model using all batches
            for k in range(0, n_train, batch_size):
                X_batch = X_loc[:, k : k + batch_size]
                Y_batch = Y_loc[:, k : k + batch_size]
                
                self.train(X_batch, Y_batch, X_batch.shape[1])

            # Val loss is the total validation loss for each batch.
            val_loss = 0
            for l in range(0, n_val, batch_size):

                X_batch = x_val[:, l : l + batch_size]
                Y_batch = y_val[:, l : l + batch_size]
                val_loss += self.evaluate(X_batch, Y_batch, self.validation_loss)

            # We compute the average val loss of the various batches.
            if batch_size < n_val:
                epoch_avg_val_loss = val_loss / (n_val / batch_size)
            else:
                epoch_avg_val_loss = val_loss

            # Setting the descent of the learning rate.    
            if eta_descent and (epoch % eta_descent_freq):
                self.eta *= 0.95
                self.update.set_eta(self.eta)

            # Setting the printing of the error, and current epoch
            if ((epoch % eta_descent_freq) == 0):
                output = self.forward_pass_model(X_loc)
                error = self.loss.forward_loss(output, Y_loc)
                print(f"Epoch: {epoch}/{epochs}")
                print(f"Error (Loss: {self.loss_name}): ", error)
            
            # If the validation error goes above the threshold more than patience times the model gets discarded.
            if (epoch_avg_val_loss - val[-1] > threshold):
                patience_count += 1
                if patience_count >= patience:
                    threshold_flag = True
                    print(f'----Bad Graph (Noisy Validation Error)----')
                    break
            
            # If the average validation loss of this epoch is better than the previous one, we update the model_best_val_loss and save the weights.
            if epoch_avg_val_loss < model_best_val_loss:
                model_best_val_loss = epoch_avg_val_loss
                data = {
                "weights": self.all_layers_weights(),
                "bias": self.all_layers_bias()
                }
                save_model(data, path_model_best_weights)

            # We compute the validation and error with different metrics to plot them.
            output_err = self.forward_pass_model(X_loc)
            output_val = self.forward_pass_model(x_val)
            for i, metric in enumerate(self.metrics):
                validation = metric.forward_loss(output_val, y_val)
                error = metric.forward_loss(output_err, Y_loc)
                err_metric[i].append(error)
                val_metric[i].append(validation)

        
            val.append(epoch_avg_val_loss) # This is still the array to keep track of the validation_loss to check the noisiness.

        if not threshold_flag:
            print(f"-------Best Val for this Configuration: {model_best_val_loss}")

        if plot:
            plot = val_err_plot()
            plot.plot(val_metric, err_metric, self.metric_names)
            
        return model_best_val_loss, threshold_flag
    
    
        
    def evaluate(self, x, y, metric):
        """
        Computes the error of the model, given the input data x, using a specific metric.
        """
        output = self.forward_pass_model(x)
        error = metric.forward_loss(output, y)
        return error
    
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
            neurons += layer.dim_output
            n_params += layer.get_weights().size + layer.get_bias().size
        
        print(f"--------------------------------------------------------------------")
        print(f"Number of neurons: {neurons};", f"Number of parameters: {n_params}.")
        print(f"Number of Mbyte: {(n_params * 8) / 1024}")
        print(f"Learning rate: {self.eta};", f"Momentum coefficient: {self.alpha}")
        print(f"--------------------------------------------------------------------")
