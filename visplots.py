import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
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

def rfDecisionPlot(XTrain, yTrain, XTest, yTest, n_estimators=10):
    plt.figure(figsize=(7,5))
    h = .02  # step size in the mesh
    Xtrain = XTrain[:, :2] # we only take the first two features.

    # Create color maps
    cmap_light = ListedColormap(["#AAAAFF", "#AAFFAA", "#FFAAAA"])
    cmap_bold  = ListedColormap(["#0000FF", "#00FF00", "#FF0000"])

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
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

def rfAvgAcc(rfModel, XTest, yTest):
    preds = []
    avgPred = []
    df = []

    for i,tree in enumerate(rfModel.estimators_):
        predTree = tree.predict(XTest)
        accTree  = round(metrics.accuracy_score(yTest, predTree),2)
        preds.append(accTree)
        if i==0:
            df = predTree
        else:
            df = np.vstack((df,predTree))


    for j in np.arange(df.shape[0]):
        j=j+1
        mv = []
        for i in np.arange(df.shape[1]):
            (values,counts) = np.unique(df[:j,i],return_counts=True)
            ind=np.argmax(counts)
            mv.append(values[ind].astype(int))
        avgPred.append(metrics.accuracy_score(yTest, mv))

    plt.figure(figsize=(10, 5))
    plt.plot(avgPred,  '.', linestyle='-', color='c')
    plt.ylim(0.8,0.95)
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()


