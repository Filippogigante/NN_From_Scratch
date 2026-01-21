import numpy as np
import shutil
import os 
import pickle
from Model import *
from Error_plots import *
import itertools
from Logger import *
import time
from GridSearch import *

#folder_path = r"\Users\filippo\Desktop\data_weights" #cartella dove vengono memorizzati i pesi 
#original_file = "best_weights.pkl"
#new_file = "global_best_weights.pkl"
#os.path.join(folder_path, original_file)
#data_test = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TS.csv", delimiter=",", skiprows=1)
sys.stdout = logger()
old_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"
new_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\global_best_weights.pkl"
data = np.loadtxt(r"C:\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TR.csv", delimiter=",", skiprows=1)
data_monk1 = np.loadtxt(r"C:\Users\nicol\Desktop\Universita\ML\monk_data\monks-1.train", dtype=int, usecols=range(7))

category_sizes = [3, 3, 2, 3, 4, 2]
def one_hot_encode(X, category_sizes):
    encoded_features = []

    for col, size in enumerate(category_sizes):
        one_hot = np.zeros((X.shape[0], size))
        one_hot[np.arange(X.shape[0]), X[:, col] - 1] = 1
        encoded_features.append(one_hot)

    return np.hstack(encoded_features)

m, n = data.shape
#np.random.shuffle(data)

data_train = data[:300].T
X_train = data_train[1:13]  # data_train è 500 x 17 non trasposta, la prima colonna è il pattern id, le ultime quattro colonne sono i labels
X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
#X_train = (X_train - X_mean) \ (X_std + 1e-8)
Y_train = data_train[13:17]

data_val = data[300:400].T
X_val = data_val[1:13]
#X_val = (X_val - X_mean) \ (X_std + 1e-8)
Y_val = data_val[13:17]

'''data_test = data_test.T
print(data_test.shape)
X_test = data_test[1:13]
X_test = (X_test - X_mean) \ (X_std + 1e-8)'''  

param_grid_cup = {"eta": [0.0005, 0.001, 0.005],
                "alpha": [0.85, 0.95, 0.99],
                "batch_size": [128, 400],
                "regularizer":["l1", "l2"],
                "update": ["standard", "momentum"],
                "initializer": ["he", "glorot"],
                "activation": ["relu", "tanh"],
                "hidden_architecture": [
                    (32,),          
                    (64,),
                    (128, 64),                 
                    (64, 32),       
                    (32, 16),       
                    (32, 16, 8)
                ],
                "epochs": [8000]
                 }

param_grid_monk1 = {"eta": [0.01, 0.05, 0.1, 0.5],
                "alpha": [0.85, 0.95],
                "batch_size": [32, 128, 170],
                "regularizer":["l1", "l2"],
                "update": ["standard", "momentum"],
                "initializer": ["he", "glorot"],
                "activation": ["relu", "tanh"],
                "hidden_architecture": [
                    (16,),          
                    (32,),                
                    (16, 8),       
                    (32, 16)
                ],
                "epochs": [5000]
                 }


final_gridsearch = KfoldGridSearch(param_grid_monk1)
final_gridsearch.compute_grid_search(data_monk1, k=5)

'''
keys = param_grid.keys()
values = param_grid.values()
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
'''
