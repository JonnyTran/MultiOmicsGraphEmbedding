import matplotlib.pyplot as plt
# Scatter plot of original graph adjacency matrix

def matrix_heatmap(data, cmap=plt.cm.bwr):
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(data, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.show()


