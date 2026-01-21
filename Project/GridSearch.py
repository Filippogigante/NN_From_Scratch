import time
from Layer import *
from Model import *
import itertools
import shutil
import math
from Logger import *
from splitdata import *

class GridSearch():

    def compute_grid_search(self):
        return NotImplementedError
    


class HoldoutGridSearch():
    
    def __init__(self,combinations):
        self.param_grid = combinations
    
    
    def compute_grid_search(self, X_train, Y_train, X_val , Y_val):
        old_path = r"\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"
        new_path = r"\Users\nicol\Desktop\Universita\ML\repo\data_weights\global_best_weights.pkl"
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))

        print(f"Inizio Grid Search su {len(combinations)} combinazioni...")
        time.sleep(2)

        best_params = None
        global_best_val_loss = float('inf')

        for combo in combinations:
            params = dict(zip(keys, combo))
            
            current_arch = params["hidden_architecture"]
            n_layers = len(current_arch)
            print(f"Configurazione {params}")

            layers =[]
            full_structure = [X_train.shape[0]] + list(current_arch) + [4] # Attacchiamo alla lista con la dimensione degli hidden layers la dimensione del primo e dell'ultimo
            for i in range(len(full_structure) - 1):
                
                if i == len(full_structure) - 2:
                    act_func = "identity" 
                else:
                    act_func = params["activation"]
                    
                layer = Layer(
                    dim_input=full_structure[i], 
                    dim_output=full_structure[i+1], 
                    activation=act_func, 
                    initializer=params["initializer"]
                )
                layers.append(layer)

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
    def __init__(self, param_grid, data_type):
        self.param_grid = param_grid
    
        split_map = {
            "cup": cup_split,        
            "monk": monk_split,
        }

        if isinstance(data_type, str):
            try:
                self.split = split_map[data_type.lower()]() 
            except KeyError:
                raise ValueError(f"Data type '{data_type}' non supportata. Usa: {list(split_map.keys())}")
        else:
            raise BaseException("data type must be a string")

    def split_dataset(self, data_training, k):
    
        result = []
        
        n_data = data_training.shape[0]
        validation_split_percentage = 100 / k
        print(f"Validation split percentage: {validation_split_percentage}")
        
        n_val_data = math.ceil(n_data / k )


        print(n_val_data)

        for i in range(k) :
            train = []
            val = data_training[(i*n_val_data) :  ((i + 1) * n_val_data)].T
            
            
            first_train = data_training[: (i*n_val_data)].T
            second_train = data_training[((i + 1) * n_val_data) : ].T
            
            print(val.shape , first_train.shape , second_train.shape)
            
            
            if i == 0:
                train = second_train
            else:
                train = np.concatenate((first_train , second_train), axis = 1)
            
            result.append((train,val))
            
        return result



    def compute_grid_search(self, data_training, k):
        
        data = self.split.split(data_training)
        old_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"
        new_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\global_best_weights.pkl"
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        global_best_val_loss = float('inf')
        counter = 0
        best_params = None
        for split in data:
            
            X_train = split[0][1:13]
            Y_train = split[0][13:17]
            X_val = split[1][1:13]
            Y_val = split[1][13:17]
            print(f"Inizio Grid Search su {len(combinations)} combinazioni..., k = {counter}")
            counter += 1
            time.sleep(1)
            
            for combo in combinations:
                    
                    params = dict(zip(keys, combo))
                
                    current_arch = params["hidden_architecture"]
                    n_layers = len(current_arch)
                    print("-----------––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––-")
                    print(f"Configurazione {params}")

                    layers =[]
                    full_structure = [X_train.shape[0]] + list(current_arch) + [4] # Attacchiamo alla lista con la dimensione degli hidden layers la dimensione del primo e dell'ultimo
                    for i in range(len(full_structure) - 1):
                        
                        if i == len(full_structure) - 2:
                            act_func = "identity" 
                        else:
                            act_func = params["activation"]
                            
                        layer = Layer(
                            dim_input=full_structure[i], 
                            dim_output=full_structure[i+1], 
                            activation=act_func, 
                            initializer=params["initializer"]
                        )
                        layers.append(layer)

                    model = Model(eta=params["eta"], alpha=params["alpha"], lamb=1e-5, layers=layers, update=params["update"], loss="mse", metric="mee", regularizer=params["regularizer"])
                    best_val_loss, Flag = model.fit(params["epochs"], X_train, Y_train, X_val, Y_val, params["batch_size"])
                    
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
                