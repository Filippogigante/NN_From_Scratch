import time
import itertools
import shutil
from layer import *
from model import *
from logger import *
from splitdata import *
from utilities import *

class GridSearch():

    def compute_grid_search(self):
        return NotImplementedError
    

class HoldoutGridSearch():
    
    def __init__(self,combinations, dataset_type):
        self.param_grid = combinations
        self.totaltime_train = 0

        dataset_type_map = {
            "cup": cup_split,        
            "monk": monk_split,
        }
        
        if isinstance(dataset_type, str):
            try:
                self.dataset = dataset_type_map[dataset_type.lower()]() 
            except KeyError:
                raise ValueError(f"Dataset '{dataset_type}' not found. Use: {list(dataset_type_map.keys())}")
        else:
            raise BaseException("activation must be a string")
    
    def compute_grid_search(self, data, val_percentage):
        
        data = self.dataset.split(data, 1, val_percentage)
        X_train , Y_train , X_val , Y_val = self.dataset.input_label_split(data[0][0] , data[0][1])
        
        old_path = r"c:\Users\franc\OneDrive\Desktop\Università\ML\NN_From_Scratch\data_weights\best_weights.pkl"
        new_path = r"c:\Users\franc\OneDrive\Desktop\Università\ML\NN_From_Scratch\data_weights\global_best_weights.pkl"
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        
        print(f"Starting grid search on {len(combinations)} combinations")

        best_params = None
        global_best_val_loss = float('inf')

        for combo in combinations:
            params = dict(zip(keys, combo))
            layers = create_layers(params, self.dataset , X_train)
            model = Model(eta=params["eta"], alpha=params["alpha"], lamb=1e-5, layers=layers, update=params["update"], loss=params["loss"], validation_loss="mse", metrics=params["metric"], regularizer=params["regularizer"])
            start = time.perf_counter()
            best_val_loss, Flag = model.fit(params["epochs"], X_train, Y_train, X_val, Y_val, params["batch_size"], plot=False)
            end = time.perf_counter()
            self.total_time_train += (end - start)
            
            if Flag:
                print("Bad graphic")
                print("\n=======================================================================================")
                continue

            if best_val_loss < global_best_val_loss:
                global_best_val_loss = best_val_loss
                best_params = params
                shutil.copy(old_path, new_path)
                print(f"--> New best configuration found! Validation Loss: {global_best_val_loss:.2f}")
                print("\n=======================================================================================")

        print("\n========================================")
        print(f"Grid search completed.")
        print(f"Average time for each model training: {self.total_time_train / len(combinations)} seconds")
        print(f"Best configuration: {best_params}")
        print(f"Best validation loss: {global_best_val_loss}")
        print("========================================")

            #print(f"Validation Error: {model.evaluate(X_val, Y_val)}")


class KfoldGridSearch():
    def __init__(self, param_grid, dataset_type):
        self.param_grid = param_grid
        
        dataset_type_map = {
            "cup": cup_split,        
            "monk": monk_split,
        }
        
        if isinstance(dataset_type, str):
            try:
                self.dataset = dataset_type_map[dataset_type.lower()]() 
            except KeyError:
                raise ValueError(f"Dataset '{dataset_type}' not found. Use: {list(dataset_type_map.keys())}")
        else:
            raise BaseException("activation must be a string")

    def compute_grid_search(self, data_training, k, val_percentage):
        
        # Split the data into training and validation
        data = self.dataset.split(data_training , k , val_percentage)
        
        old_path = r"c:\Users\franc\OneDrive\Desktop\Università\ML\NN_From_Scratch\data_weights\best_weights.pkl"
        new_path = r"c:\Users\franc\OneDrive\Desktop\Università\ML\NN_From_Scratch\data_weights\global_best_weights.pkl"
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        global_best_val_loss = float('inf')
        counter = 0
        best_params = None
        
        for split in data:
            
            total_time_kfold = 0
            #split each training and validation into input and corresponding label
            X_train,Y_train,X_val,Y_val  = self.dataset.input_label_split(split[0], split[1])
            
            print(f"Starting grid search on {len(combinations)} combinations, k = {counter}")
            counter += 1
            
            for combo in combinations:
   
                params = dict(zip(keys, combo))
                layers = create_layers(params, self.dataset , X_train)
                model = Model(eta=params["eta"], alpha=params["alpha"], lamb=1e-5, layers=layers, update=params["update"], loss=params["loss"], validation_loss="mse", metrics=params["metric"], regularizer=params["regularizer"])
                start = time.perf_counter()
                model_best_val_loss, threshold_flag = model.fit(params["epochs"], X_train, Y_train, X_val, Y_val, params["batch_size"], plot=False)
                end = time.perf_counter()
                total_time_kfold += end - start

                # If the current model had a noisy validation trend, it gets discarded. We pass to the next comfiguration.
                if threshold_flag:
                    print("\n=======================================================================================")
                    continue

                # If the current model's best validation loss is the best one, we save this model weights and update the global_best_val_loss.
                if model_best_val_loss < global_best_val_loss:
                    global_best_val_loss = model_best_val_loss
                    best_params = params
                    shutil.copy(old_path, new_path)
                    print("\n=======================================================================================")
                    print(f"--> New best configuration found! Validation Loss: {global_best_val_loss:.2f}")
                    print("\n=======================================================================================")

            print(f"Total time for this fold (k = {counter}): {total_time_kfold} seconds")
            print(f"Average time for each model training: {total_time_kfold / len(combinations)} seconds")
            print("\n========================================")
            print(f"Grid search completed.")
            print(f"Best configuration: {best_params}")
            print(f"Best validation loss: {global_best_val_loss}")
            print("========================================")
            