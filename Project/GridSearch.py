import time
from Layer import *
from Model import *
import itertools
import shutil
import math
from Logger import *
from splitdata import *
from utilities import *

class GridSearch():

    def compute_grid_search(self):
        return NotImplementedError
    


class HoldoutGridSearch():
    
    def __init__(self,combinations, dataset_type):
        self.param_grid = combinations
        
        dataset_type_map = {
            "cup": cup_split,        
            "monk": monk_split,
        }
        
        if isinstance(dataset_type, str):
            try:
                self.dataset = dataset_type_map[dataset_type.lower()]() 
            except KeyError:
                raise ValueError(f"Dataset '{dataset_type}' non trovato. Usa: {list(dataset_type_map.keys())}")
        else:
            raise BaseException("activation must be a string")
    
    
    def compute_grid_search(self, data , k , val_percentage):
        
        data = self.dataset.split(data, k , val_percentage)
        X_train , Y_train , X_val , Y_val = self.dataset.input_label_split(data[0] , data[1])
        
        old_path = r"\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"
        new_path = r"\Users\nicol\Desktop\Universita\ML\repo\data_weights\global_best_weights.pkl"
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        params = dict(zip(keys, combo))
        
        print(f"Inizio Grid Search su {len(combinations)} combinazioni...")
        time.sleep(2)

        best_params = None
        global_best_val_loss = float('inf')

        for combo in combinations:
            params = dict(zip(keys, combo))
            layers = create_layers(params, self.dataset , X_train)
            model = Model(eta=params["eta"], alpha=params["alpha"], lamb=1e-5, layers=layers, update=params["update"], loss="mse", metric="mee", regularizer="l1")
            best_val_loss, Flag = model.fit(params["epochs"], X_train, Y_train, X_val, Y_val, params["batch_size"])
            
            if Flag:
                print("Il grafico è venuto male")
                print("\n=======================================================================================")
                continue

            if best_val_loss < global_best_val_loss:
                global_best_val_loss = best_val_loss
                best_params = params
                shutil.copy(old_path, new_path)
                print(f"--> Nuova configurazione migliore trovata! Validation Loss: {global_best_val_loss:.2f}")
                print("\n=======================================================================================")

        print("\n========================================")
        print(f"Grid Search Completata.")
        print(f"Migliore Configurazione: {best_params}")
        print(f"Migliore Validation Loss: {global_best_val_loss}")
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
                raise ValueError(f"Dataset '{dataset_type}' non trovato. Usa: {list(dataset_type_map.keys())}")
        else:
            raise BaseException("activation must be a string")



    def compute_grid_search(self, data_training, k, val_percentage):
        
        #split the data into training and validation
        data = self.dataset.split(data_training , k , val_percentage)
        
        old_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"
        new_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\global_best_weights.pkl"
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        global_best_val_loss = float('inf')
        counter = 0
        best_params = None
        
        for split in data:
            
            #split each training and validation into input and corresponding label
            X_train,Y_train,X_val,Y_val  = self.dataset.input_label_split(split[0], split[1])
            
            print(f"Inizio Grid Search su {len(combinations)} combinazioni..., k = {counter}")
            counter += 1
            time.sleep(1)
            
            for combo in combinations:
                    
                    
                    params = dict(zip(keys, combo))

                    current_arch = params["hidden_architecture"]
                    layers = create_layers(params, self.dataset , X_train)
                    model = Model(eta=params["eta"], alpha=params["alpha"], lamb=1e-5, layers=layers, update=params["update"], loss=params["loss"], metric=params["metric"], regularizer=params["regularizer"])
                    best_val_loss, Flag = model.fit(params["epochs"], X_train, Y_train, X_val, Y_val, params["batch_size"], plot = True)
                    
                    if Flag:
                        print("\n=======================================================================================")
                        continue

                    if best_val_loss < global_best_val_loss:
                        global_best_val_loss = best_val_loss
                        best_params = params
                        shutil.copy(old_path, new_path)
                        print("\n=======================================================================================")
                        print(f"--> Nuova configurazione migliore trovata! Validation Loss: {global_best_val_loss:.2f}")
                        print("\n=======================================================================================")
                        
                    

            print("\n========================================")
            print(f"Grid Search Completata.")
            print(f"Migliore Configurazione: {best_params}")
            print(f"Migliore Validation Loss: {global_best_val_loss}")
            print("========================================")
                