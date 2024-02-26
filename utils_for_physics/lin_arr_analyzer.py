from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class CreateArrays:

    @staticmethod
    def _basic_f1(x, al1, al2):
        return al1 * np.sin(al2 * x)

    @staticmethod
    def _basic_f2(x, al1, al2):
        return al1 * np.exp(-al2 * x)

    @staticmethod
    def _basic_f3(x, al1, al2):
        return al1 * np.exp(-al2 * x)

    @staticmethod
    def _basic_f4(x, al1, al2):
        return al1 * x + al2

    @staticmethod
    def _basic_f5(x, al1, al2, al3):
        return al1 * x * x + al2 * x + al3


    def __init__(self, xmin1, xmax1, xmin2, xmax2, smin, smax, delmin, delmax, n):
        self.xmin1 = xmin1
        self.xmax1 = xmax1
        self.xmin2 = xmin2
        self.xmax2 = xmax2
        self.smin = smin
        self.smax = smax
        self.delmin = delmin
        self.delmax = delmax
        self.n = n


    def _create_x_array(self) -> tuple[np.array, np.array, np.array, np.array]:
        x1 = np.random.random([1, self.n])
        x1 = self.xmin1 + (self.xmax1 - self.xmin1) * x1
        x2 = np.random.random([1, self.n])
        x2 = self.xmin2 + (self.xmax2 - self.xmin2) * x2
        x3 = np.random.random([1, self.n])
        x4 = np.random.random([1, self.n])

        return np.sort(x1), np.sort(x2), np.sort(x3), np.sort(x4)


    def _create_y_array(self) -> tuple[np.array, np.array]:
        x1, x2, x3, x4 = self._create_x_array()
        y1 = self._basic_f5(x=x1, al1=0.1, al2=3, al3=5) * (1. + self.delmin + self.delmax * x3)
        y2 = self._basic_f5(x=x2, al1=0.1, al2=3, al3=5) * (1. + self.delmin + self.delmax * x4)

        return y1, y2


    def _create_dy_array(self) -> tuple[np.array, np.array]:
        y1, y2 = self._create_y_array()
        dy1 = 0.05 * abs(y1)
        dy2 = 0.15 * abs(y2)

        return dy1, dy2


class ArraysAnalyzer:
    def __init__(self, x_train, y_train, x_test, dy_low, dy_high):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.dy_low = dy_low
        self.dy_high = dy_high


    def plot_x_and_y(self):
        plt.figure(figsize=(12,12))
        sns.lineplot(x=self.x_train, y=self.y_train, palette='rainbow')
        plt.grid(True)
        plt.xlabel('X', fontsize=15)
        plt.ylabel('Y', fontsize=15)
        plt.title('X and y dependency graph', fontsize=15)
        plt.show();
    

    def plot_dy(self):
        plt.figure(figsize=(12,12))
        plt.subplot(1,1,2)
        sns.histplot(data=self.dy_low, palette='rainbow')
        plt.xlabel('dy')
        plt.title('dy low histogram')
        plt.subplot(1,2,2)
        sns.histplot(data=self.dy_high, palette='rainbow')
        plt.title('dy high histogram')
        plt.xlabel('dy')
        plt.show();


    def _scale_data(self):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(self.x_train)
        return x_scaled
    

    @staticmethod
    def _chi_squared_loss(y_true: np.array, y_pred: np.array, dy: np.array) -> np.array:
        loss_chi = np.abs(y_true - y_pred) / np.abs(dy)
        return np.mean(loss_chi ** 2)


    def linear_model(self, scale: bool = False) -> Optional[LinearRegression]:
        try:
            if scale:
                self.x_train = self._scale_data()
            lin_reg = LinearRegression()
            lin_reg.fit(self.x_train, self.y_train)
            return lin_reg
        except Exception as e:
            print('Something went wrong')
            raise e
    
    def _plot_linear_model(self, y_true: np.array, y_pred: np.array, dy_low: np.array, dy_high: np.array) -> None:
        chi_squared_low = self._chi_squared_loss(y_true, y_pred, dy_low)
        chi_squared_high = self._chi_squared_loss(y_true, y_pred, dy_high)

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
        plt.show();


    def linear_model_results(self, lin_reg_model: LinearRegression, y_true: np.array) -> None:
        y_pred = lin_reg_model.predict(self.x_test)
        coefficents = lin_reg_model.coef_
        print(pd.DataFrame(coefficents))
        self._plot_linear_model(y_true, y_pred, self.dy_low, self.dy_high)
