import matplotlib.pyplot as plt
import scipy.sparse as sp

def plot_matrix(array, figsize=(10, 10), aspect=None, cmap=plt.cm.gray, colorbar=True):
    if type(array) is sp.csr.csr_matrix:
        array = array.todense()

    plt.figure(figsize=figsize)
    plt.imshow(array, interpolation='none', cmap=cmap, aspect=aspect)
    if colorbar:
        plt.colorbar()
    plt.show()
