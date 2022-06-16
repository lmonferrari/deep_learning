import numpy as np

from entrada import Input
from linear import Linear
from sigmoide import Sigmoid
from custo import MSE
from topological_sort import topological_sort, forward_and_backward, sgd_update
from pre_processamento import normalize

from sklearn.datasets import fetch_california_housing
from sklearn.utils import resample, shuffle


if __name__ == '__main__':

    # load the data
    data = fetch_california_housing()

    # spliting into features and target
    X_ = data['data']
    y_ = data['target']

    # normalizing data
    X_ = normalize(X_)

    # number of features and number of neurons
    n_features = X_.shape[1]
    n_hidden = 10

    # initializing weights and bias
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, 1)
    b2_ = np.zeros(1)

    # Neural network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()

    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)

    # Feed_dict definition
    feed_dict = {
        X: X_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_,
        y: y_
    }

    # epoch size
    epochs = 600

    # total lines/records
    m = X_.shape[0]

    # batch size
    batch_size = 11
    steps_per_epoch = m // batch_size

    # Sort the input for execution
    graph = topological_sort(feed_dict)

    params = [W1, b1, W2, b2]

    print(f'Total samples: {m}')

    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

            X.value = X_batch
            y.value = y_batch

            forward_and_backward(graph)

            sgd_update(params)

            loss += graph[-1].value

        print(f'Epoch {i + 1} - Cost {loss/steps_per_epoch}')
