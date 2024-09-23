import os

import numpy as np
from sklearn.linear_model import LinearRegression


class AllreduceModel(object):
    def __init__(self):
        self.a = 0
        self.b = 0

    def train(self, x_data, y_data):

        x = np.array(x_data).reshape((-1, 1))
        y = np.array(y_data)

        # train model
        model = LinearRegression()
        model.fit(x, y)

        self.a = model.coef_[0]
        self.b = model.intercept_

    def evaluate(self, x):
        """param x:float - theoretical all-reduce time w.r.t bucket size"""
        return self.a*x + self.b

    def save(self, train_result_file_path='train_result/model_params.txt'):
        os.makedirs(os.path.dirname(train_result_file_path), exist_ok=True)
        with open(train_result_file_path, 'w') as f:
            f.write(f'{self.a},{self.b}')

    def load(self, param_file_path):
        try:
            with open(param_file_path, 'r') as f:
                line = f.readline()
                self.a = float(line.split(',')[0])
                self.b = float(line.split(',')[1])
        except Exception as e:
            print(e)
            print(f'[ERROR] file path: {param_file_path}')
        print(f'Load Model parameters => a: {self.a} | b: {self.b}')

    def __repr__(self):
        return f'a: {self.a} | b: {self.b}'
