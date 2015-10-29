import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def knnDecisionPlot(XTrain, yTrain, XTest, yTest, n_neighbors, weights):
    plt.figure(figsize=(7,5))
    h = .02  # step size in the mesh
    Xtrain = XTrain[:, :2] # we only take the first two features.

    # Create color maps
    cmap_light = ListedColormap(["#AAAAFF", "#AAFFAA", "#FFAAAA"])
    cmap_bold  = ListedColormap(["#0000FF", "#00FF00", "#FF0000"])

    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(Xtrain, yTrain)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    plt.scatter(XTest[:, 0], XTest[:, 1], c = yTest, cmap = cmap_bold)
    plt.contour(xx, yy, Z, colors=['k'], linestyles=['-'], levels=[0])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('fixed_acidity')
    plt.ylabel('volatile_acidity')
    plt.title("2-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.show()

def rfDecisionPlot(XTrain, yTrain, XTest, yTest):
    plt.figure(figsize=(7,5))
    h = .02  # step size in the mesh
    Xtrain = XTrain[:, :2] # we only take the first two features.

    # Create color maps
    cmap_light = ListedColormap(["#AAAAFF", "#AAFFAA", "#FFAAAA"])
    cmap_bold  = ListedColormap(["#0000FF", "#00FF00", "#FF0000"])

    clf = RandomForestClassifier()
    clf.fit(Xtrain, yTrain)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    plt.scatter(XTest[:, 0], XTest[:, 1], c = yTest, cmap = cmap_bold)
    plt.contour(xx, yy, Z, colors=['k'], linestyles=['-'], levels=[0])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('fixed_acidity')
    plt.ylabel('volatile_acidity')
    plt.title("2-Class classification Random Forests")
    plt.show()

