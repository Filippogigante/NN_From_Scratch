import numpy as np
import pickle
from Model import *
from Error_plots import *
from Dropout import *

#data = np.loadtxt(r"\Users/nicol/Desktop/Universita/ML/cup_data/ML-CUP25-TR.csv", delimiter=",", skiprows=1)
#data_test = np.loadtxt(r"/Users/nicol/Desktop/Universita/ML/cup_data/ML-CUP25-TS.csv", delimiter=",", skiprows=1)
#data_monk = np.loadtxt(r"/Users/filippo/desktop/NiralNeuorcFromScretch-jaeger/datasets/monks-1.train", dtype=int, usecols=range(7))


data_monk = np.loadtxt(r"/Users/filippo/desktop/NiralNeuorcFromScretch-jaeger/datasets/monks-2.train", dtype=int, usecols=range(7))


## Monk-managemenent
category_sizes_monk2 = [3, 3, 2, 3, 3, 2]
category_sizes = [3, 3, 2, 3, 4, 2]
def one_hot_encode(X, category_sizes):
    encoded_features = []

    for col, size in enumerate(category_sizes):
        one_hot = np.zeros((X.shape[0], size))
        one_hot[np.arange(X.shape[0]), X[:, col] - 1] = 1
        encoded_features.append(one_hot)

    return np.hstack(encoded_features)


#preprocessing monk1
"""
Y_monk = data_monk[:, 0]          # labels
X_monk = data_monk[:, 1:]       # attributes

X_encoded = one_hot_encode(X_monk, category_sizes)
Y_monk = Y_monk.reshape(-1, 1)
Y_monk = Y_monk.astype(int)

# Shuffle the dataset
indices = np.random.permutation(X_encoded.shape[0])
X_encoded = X_encoded[indices]
Y_monk = Y_monk[indices]

split = int(0.8 * X_encoded.shape[0])

X_train_monk, X_val_monk = X_encoded[:split], X_encoded[split:]
Y_train_monk, Y_val_monk = Y_monk[:split], Y_monk[split:]

X_train_monk = X_train_monk.T
Y_train_monk = Y_train_monk.T
X_val_monk = X_val_monk.T
Y_val_monk = Y_val_monk.T
"""
#preprocessing monk2

Y_monk = data_monk[:, 0]          # labels
X_monk = data_monk[:, 1:]       # attributes

X_encoded = one_hot_encode(X_monk, category_sizes)
Y_monk = Y_monk.reshape(-1, 1)
Y_monk = Y_monk.astype(int)

# Shuffle the dataset
indices = np.random.permutation(X_encoded.shape[0])
X_encoded = X_encoded[indices]
Y_monk = Y_monk[indices]

split = int(0.8 * X_encoded.shape[0])

X_train_monk, X_val_monk = X_encoded[:split], X_encoded[split:]
Y_train_monk, Y_val_monk = Y_monk[:split], Y_monk[split:]

X_train_monk = X_train_monk.T
Y_train_monk = Y_train_monk.T
X_val_monk = X_val_monk.T
Y_val_monk = Y_val_monk.T


## Cup-management
"""
m, n = data.shape
#np.random.shuffle(data)

data_train = data[:300].T
X_train = data_train[1:13]  # data_train è 500 x 17 non trasposta, la prima colonna è il pattern id, le ultime quattro colonne sono i labels
X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
#X_train = (X_train - X_mean) \ (X_std + 1e-8)
Y_train = data_train[13:17]


data_test = data[300:400].T
X_test = data_test[1:13]
#X_test = (X_test - X_mean) \ (X_std + 1e-8)
Y_test = data_test[13:17]



data_test_real = data[400:].T
X_test_real = data_test_real[1:13]
#X_test = (X_test - X_mean) \ (X_std + 1e-8)
Y_test_real = data_test_real[13:17]
"""
'''data_test = data_test.T
print(data_test.shape)
X_test = data_test[1:13]
X_test = (X_test - X_mean) \ (X_std + 1e-8)'''  

#layers = [Layer(X_train.shape[0], 32, "relu", "glorot"), Layer(32, 16, "relu", "glorot"), Layer(16, 4, "identity", "glorot")]
#model = Model(eta=0.0001, alpha=0.8, lamb=1e-5, layers=layers, update="momentum", loss="mse", metric="mee", regularizer="l1")
#model.fit(12000, X_train, Y_train, X_test, Y_test, batch_size=128)

print("----------------------------------------------------------")

#Carichiamo i pesi 


# Define the path to your weights
#weights_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\global_best_weights.pkl"
#best_weights_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\best_weights.pkl"

# --- Load the pickle files ---
#with open(weights_path, 'rb') as f:
    #data_loaded = pickle.load(f)

"""
with open(best_weights_path, 'rb') as f:
    best_data_loaded = pickle.load(f)
"""

# --- Extract weights and biases ---
#weights = data_loaded['weights']
#biases = data_loaded['bias']

#best_weights = best_data_loaded['weights']
#best_biases = best_data_loaded['bias']

# --- Convert each layer to NumPy arrays to preserve structure ---
#weights = [np.array(w) for w in weights]
#biases = [np.array(b) for b in biases]

#best_weights = [np.array(w) for w in best_weights]
#best_biases = [np.array(b) for b in best_biases]


"""
same = all(
    np.allclose(w, bw, rtol=1e-9, atol=1e-9)
    for w, bw in zip(weights, best_weights)
)

print("Weights identical (within tolerance):", same)
"""

layers_1 = [Layer(X_train_monk.shape[0], 16, "tanh", "glorot"), Layer(16, 1, "sigmoid", "glorot") ]
#carico i pesi nei layer del modello 1

"""
for w,layer in zip(weights,layers_1):
    layer.set_weights(w)
    
for b,layer in zip(biases, layers_1):
    layer.set_bias(b)
    
for i, layer in enumerate(layers_1):
    print(f"Layer {i}: W {layer.W.shape}, b {layer.b.shape}")

"""
    
"""
layers_2 = [Layer(X_train.shape[0], 32, "relu", "glorot"), Layer(32, 16, "relu", "glorot"), Layer(16, 4, "identity", "glorot")]
for w,layer in zip(best_weights,layers_2):
    layer.set_weights(w)
    
for b,layer in zip(best_biases, layers_2):
    layer.set_bias(b)

"""
model_1 = Model(eta=1, alpha=0.95, lamb=1e-5, layers=layers_1, update="momentum", loss="binary_cross_entropy", metric="accuracy", regularizer="l1")

model_1.fit(10000, X_train_monk, Y_train_monk, X_val_monk, Y_val_monk, 400, plot=True)
#model_2 = Model(eta=0.0001, alpha=0.8, lamb=1e-5, layers=layers_2, update="momentum", loss="mse", metric="mee", regularizer="l1")

#carico i pesi nel modello 1

#print(model_1.evaluate(X_test_real,Y_test_real))
#print(model_2.evaluate(X_test_real,Y_test_real))
