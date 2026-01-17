import time
from Layer import *
from Model import *
import itertools
import shutil
import math
from Logger import *


sys.stdout = logger()
data = np.loadtxt(r"/Users/filippo/desktop/NiralNeuorcFromScretch-jaeger/Project/ML-CUP25-TR.csv", delimiter=",", skiprows=1)


m, n = data.shape
#np.random.shuffle(data)

data_train = data[:300].T
X_train = data_train[1:13]  # data_train è 500 x 17 non trasposta, la prima colonna è il pattern id, le ultime quattro colonne sono i labels
X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
#X_train = (X_train - X_mean) / (X_std + 1e-8)
Y_train = data_train[13:17]

data_val = data[300:400].T
X_val = data_val[1:13]
#X_val = (X_val - X_mean) / (X_std + 1e-8)
Y_val = data_val[13:17]


class GridSearch():

    def compute_grid_search(self):
        return NotImplementedError
    


class HoldoutGridSearch():
    
    def __init__(self,combinations):
        self.param_grid = combinations
    
    
    def compute_grid_search(self, X_train, Y_train, X_val , Y_val):
        old_path = r"/Users/filippo/desktop/NiralNeuorcFromScretch-jaeger/data_weights/best_weights.pkl"
        new_path = r"/Users/filippo/desktop/NiralNeuorcFromScretch-jaeger/data_weights/global_best_weights.pkl"
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
    def __init__(self, param_grid):
        self.param_grid = param_grid
    
    def split_dataset(self, data_training, k ):
    
        result = []
        
        n_data = data_training.shape[0]
        
        validation_split = (n_data / k)
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
                train = np.concatenate((first_train , second_train), axis = 1 )
            
            result.append((train,val))
            
        return result



    def compute_grid_search(self, data_training , k ):
        
        data = self.split_dataset(data_training , k)
        old_path = r"/Users/filippo/desktop/NiralNeuorcFromScretch-jaeger/data_weights/best_weights.pkl"
        new_path = r"/Users/filippo/desktop/NiralNeuorcFromScretch-jaeger/data_weights/global_best_weights.pkl"
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        global_best_val_loss = float('inf')
        counter = 0 
        for split in data:
            
            X_train = split[0][1:13]
            Y_train = split[0][13:17]
            X_val = split[1][1:13]
            Y_val = split[1][13:17]
            print(f"Inizio Grid Search su {len(combinations)} combinazioni..., k = {counter}")
            counter += 1
            time.sleep(1)
            best_params = None
            
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
                
            
        
        
        
                
"""
param_grid = {"eta": [0.0001],
                "alpha": [0.80],
                "batch_size": [400, 128],
                "update": ["momentum"],
                "initializer": ["glorot"],
                "activation": ["relu"],
                "hidden_architecture": [
                    (32,),      
                    (32, 16),       
                    #(64, 32, 16)
                ],
                "epochs": [12000]
                 }
                 """

param_grid = {"eta": [0.0001],
                "alpha": [0.80],
                "batch_size": [400, 128],
                "update": ["momentum"],
                "initializer": ["glorot"],
                "activation": ["relu"],
                "hidden_architecture": [
                    (32,),      
                    (32, 16),       
                    #(64, 32, 16)
                ],
                "epochs": [6000]
                 }

    
a = KfoldGridSearch(param_grid)


a.compute_grid_search(data, 4)