import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

print("Download datasets ....")
digits_train = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
    header=None)
digits_test = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
    header=None)
print("Datasets downloaded!")

X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)


def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.to_numpy() == i]
        py = X_pca[:, 1][y_digits.to_numpy() == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component!')
    plt.ylabel('Second Principal Component!')
    plt.show()


plot_pca_scatter()
