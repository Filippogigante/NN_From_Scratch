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
#old_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"
#new_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\global_best_weights.pkl"
#data = np.loadtxt(r"C:\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TR.csv", delimiter=",", skiprows=1)
data_monk1 = np.loadtxt("/Users/Filippo/Desktop/NiralNeuorcFromScretch-jaeger/datasets/monks-2.train", dtype=int, usecols=range(7))


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
"""
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
"""

param_grid_monk1 = {"eta": [0.1],
                "loss" : ["binary_cross_entropy"],
                "metric" : ["error"],
                "alpha": [0.95],
                "batch_size": [ 128],
                "regularizer":["l1"],
                "update": ["momentum"],
                "initializer": ["glorot"],
                "activation": ["relu",],
                "hidden_architecture": [
                    (8,),                
                ],
                "epochs": [2000]
                 }


final_gridsearch = KfoldGridSearch(param_grid_monk1, "monk")
final_gridsearch.compute_grid_search(data_monk1, k=5 , val_percentage= 0.2)


