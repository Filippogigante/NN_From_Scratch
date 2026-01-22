from layer import *
from model import *
import pickle
from logger import *
from splitdata import *

def create_layers(params, dataset, X_train):

    current_arch = params["hidden_architecture"]
    act_func = ""
    last_layer_size = None
    
    if isinstance(dataset , cup_split):
        act_func = "identity" 
        last_layer_size = 4
    elif isinstance(dataset , monk_split):
        act_func = "sigmoid"
        last_layer_size = 1
                
    print("-------------------------------------------------------------")
    print(f"Configurazione {params}")
    print("-------------------------------------------------------------")
    
    layers =[]
    
    full_structure = [X_train.shape[0]] + list(current_arch) + [last_layer_size] # Attacchiamo alla lista con la dimensione degli hidden layers la dimensione del primo e dell'ultimo
    
    for i in range(len(full_structure) - 1):
        
        if i == len(full_structure) - 2:
            if isinstance(dataset , cup_split):
                act_func = "identity" 
            elif isinstance(dataset , monk_split):
                act_func = "sigmoid" 
            else:
                return TypeError("self.dataset is neither cup nor monk")
        else:
            act_func = params["activation"]
            
        layer = Layer(
            dim_input=full_structure[i], 
            dim_output=full_structure[i+1], 
            activation=act_func, 
            initializer=params["initializer"]
        )
        layers.append(layer)
          
    return layers

def save_model(data, path):

    with open(path, 'wb') as file:  
        pickle.dump(data, file)