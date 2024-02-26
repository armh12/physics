import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_numpy_array_dat(doc: str) -> np.array:
    data = pd.read_csv(doc, delimiter = '\t', header = None)
    arr = data.values
    arr_1 = []

    for row in arr:
        for elem in row:
            one_row = elem.split(' ')
            one_row = np.array(
                list(
                    filter(lambda x: x != '', one_row)
                    ),
            dtype = np.float32)
            arr_1.append(one_row)   
          
    return np.array(arr_1).flatten()

def chi_squared(y_true: np.array, y_pred: np.array, dy: np.array) -> float:
    loss_chi = np.abs(y_true - y_pred) / np.abs(dy)
    return np.mean(loss_chi ** 2)

def plot_linear(y_true, y_pred, chi_squared_low, chi_squared_high):
    plt.figure(figsize = (12, 8))
    label = f'''
Chi Squared low - {chi_squared_low:.3f}
Chi Squared high - {chi_squared_high:.3f}
    '''
    plt.scatter(y_true, y_pred, color = 'red', marker = 'o')
    plt.scatter(y_pred, y_pred, color = 'blue', marker = 'o')
    plt.plot(y_true, y_true, color = 'green', label = label)
    plt.title('Linear Regression for 100 species validation split results')
    plt.legend(loc = 'upper left')

def make_numpy_array_1000(doc: str) -> np.array:
    data = pd.read_csv(doc)
    return data.values.flatten()