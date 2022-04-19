import pandas as pd
import numpy as np
from model.perceptron import Perceptron
import matplotlib.pyplot as plt


if __name__ == '__main__':

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(url, header=None)

    y = df.iloc[0:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    perceptron_clf = Perceptron(0.01, 10)
    perceptron_clf.train(X, y)
    print(perceptron_clf.w)

    plt.title('Perceptron')
    plt.xlabel('Iter')
    plt.ylabel('Classification errors')
    plt.plot(range(1, len(perceptron_clf.errors)+1), perceptron_clf.errors, marker='o')
    plt.show()
