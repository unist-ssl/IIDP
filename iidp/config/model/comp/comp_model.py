import numpy as np


class VSWThroughputModel(object):
    def __init__(self):
        self.a = 0
        self.b = 0
        self.x_data = []
        self.y_data = []

    def train(self, x_data, y_data):
        """
        Arguments:
            x_data: List of number of VSWs
            y_data: List of throughput normalized by 1 VSW
        """
        self.x_data = x_data
        self.y_data = y_data
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        xlog_data = np.log(x_data)
        self.a, self.b = np.polyfit(xlog_data, y_data, 1)

    def evaluate(self, x):
        """
        Arguments:
            x: Number of VSWs
        Return:
            Normalized throughput
        """
        return self.a * np.log(x) + self.b

    def __repr__(self):
        pass


class VSWRealThroughputModel(object):
    def __init__(self):
        self.true_thp_model = []

    def train(self, x_data, y_data):
        self.true_thp_model = y_data

    def evaluate(self, x):
        return self.true_thp_model[x-1]