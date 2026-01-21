import numpy as np
import math

class split_my_data:

    def split(self, data, k):
        return NotImplementedError
    
class cup_split(split_my_data):

    def split(self, data, k , val_percentage):
        result = []
        n_data = data.shape[0]
        n_val_data = math.ceil(n_data * val_percentage )
        
        if ( 1 / k) != (val_percentage): 
            print("Attenzione: il validation set nelle k-fold non copre tutti i dati")
            
            
        for i in range(k) :
            train = []
            val = data[(i*n_val_data) :  ((i + 1) * n_val_data)].T
            
            
            first_train = data[: (i*n_val_data)].T
            second_train = data[((i + 1) * n_val_data) : ].T
            
            print(val.shape , first_train.shape , second_train.shape)
            
            
            if i == 0:
                train = second_train
            else:
                train = np.concatenate((first_train , second_train), axis = 1)
            
            result.append((train,val))
            
        return result
    
    def input_label_split(self,train,val):
        X_train = train[1:13]
        Y_train = train[13:17]
        X_val = val[1:13]
        Y_val = val[13:17]
        
        return X_train , Y_train , X_val , Y_val
        
class monk_split:
    
    #restituisce un'array di tuple: ogni tupla è un (train,val) split di una fold
    def split(self, data, k , val_percentage):
        result = []
        n_data = data.shape[0]

        if ( 1 / k) != (val_percentage): 
            print("Attenzione: il validation set nelle k-fold non copre tutti i dati")


        n_val_data = math.ceil(n_data * val_percentage )

        category_sizes = [3, 3, 2, 3, 4, 2]
        Y_monk = data[:, 0]          # labels
        X_monk = data[:, 1:]       # attributes
        X_encoded = self.one_hot_encode(X_monk, category_sizes)
        Y_monk = Y_monk.reshape(-1, 1)
        Y_monk = Y_monk.astype(int)
        # Shuffle the dataset
        indices = np.random.permutation(X_encoded.shape[0])
        X_encoded = X_encoded[indices]
        Y_monk = Y_monk[indices]
        data = np.concatenate((Y_monk , X_encoded), axis=1)

        for i in range(k) :
            train = []
            val = data[(i*n_val_data) :  ((i + 1) * n_val_data)].T
            first_train = data[: (i*n_val_data)].T
            second_train = data[((i + 1) * n_val_data) : ].T
            
            
            
            
            if i == 0:
                train = second_train
            else:
                train = np.concatenate((first_train , second_train), axis = 1)
            
            result.append((train,val))
        return result

    def one_hot_encode(self, X, category_sizes):
            encoded_features = []

            for col, size in enumerate(category_sizes):
                one_hot = np.zeros((X.shape[0], size))
                one_hot[np.arange(X.shape[0]), X[:, col] - 1] = 1
                encoded_features.append(one_hot)

            return np.hstack(encoded_features)
    
    def input_label_split(self,train,val):
        X_train = train[1:17]
        Y_train = train[17:18]
        X_val = val[1:17]
        Y_val = val[17:18]
        
        return X_train , Y_train , X_val , Y_val 
        


"""
data_monk = np.loadtxt("/Users/Filippo/Desktop/NiralNeuorcFromScretch-jaeger/datasets/monks-1.train", dtype=int, usecols=range(7))

splitter = monk_split()

splitting = splitter.split(data_monk , 5)

print(len(splitting))

x_y_split = splitter.input_label_split(splitting[0][0], splitting[0][1])

print(x_y_split[0].shape , x_y_split[1].shape)
"""
