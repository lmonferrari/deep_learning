import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.adaline import AdalineGD


if __name__ == '__main__':
    adagd = AdalineGD(learning_rate=0.00001, epochs=80)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(url, header=None)

    y = df.iloc[0:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    adagd.train(X, y)

    plt.plot(range(1, len(adagd.cost) + 1), adagd.cost, marker='o')
    plt.title('Adaline')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

