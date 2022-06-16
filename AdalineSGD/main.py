import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.adaline import AdalineSGD


if __name__ == '__main__':

    adasgd = AdalineSGD(learning_rate=0.0001, epochs=50)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(url, header=None)

    y = df.iloc[0:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    adasgd.train(X, y)

    plt.plot(range(1, len(adasgd.cost) + 1), adasgd.cost, marker='o')
    plt.title('Adaline SGD')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
