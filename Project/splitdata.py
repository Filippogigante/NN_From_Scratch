import numpy as np
import math

class split_my_data:

    def split(self, data, k):
        return NotImplementedError
    
class cup_split(split_my_data):

    def split(self, data, k):
        result = []
        
        n_data = data.shape[0]
        validation_split_percentage = 100 / k
        print(f"Validation split percentage: {validation_split_percentage}")
        
        n_val_data = math.ceil(n_data / k )

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
    
class monk_split:
    
    def split(self, data, k):
        result = []
        n_data = data.shape[0]
        validation_split_percentage = 100 / k
        n_val_data = math.ceil(n_data / k )
        print(f"Validation split percentage: {validation_split_percentage}")
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
            
            print(val.shape , first_train.shape , second_train.shape)
            
            
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