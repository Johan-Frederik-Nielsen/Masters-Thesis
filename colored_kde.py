import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    arr = np.concatenate((np.linspace(0, 1, 100), np.linspace(-1, 1, 10)))
    sns.kdeplot(arr, color='green', label='green')
    arr = np.concatenate((np.linspace(-1, 0, 100), np.linspace(-1, 1, 10)))
    sns.kdeplot(arr, color='red', label='red')
    plt.legend()
    plt.savefig('color_test.png')
