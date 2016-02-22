import numpy as np
import plotly.graph_objs as go

from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def knnDecisionPlot(XTrain, yTrain, XTest, yTest, header, n_neighbors, weights = "uniform"):
    Xtrain = XTrain[:, :2]
    h = .02  # step size in the mesh

    clf = KNeighborsClassifier(n_neighbors, weights)
    clf.fit(Xtrain, yTrain)

    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    trace1 = go.Contour(
        x = np.arange(x_min, x_max, h),
        y = np.arange(y_min, y_max, h),
        z = Z.reshape(xx.shape),
        showscale=False,
        opacity=0.8,
        line = dict(
            width = 1,
            color = 'black'
        ),
        colorscale=[[0, '#AAAAFF'], [1, '#FFAAAA']],  # custom colorscale
    )

    trace2 = go.Scatter(
        x = XTest[yTest == 0,0],
        y = XTest[yTest == 0,1],
        mode = 'markers',
        marker = Marker(
            color = '#0000FF',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'high quality (test)'
    )

    trace3 = go.Scatter(
        x = XTest[yTest == 1,0],
        y = XTest[yTest == 1,1],
        mode = 'markers',
        marker = Marker(
            color = '#FF0000',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'low quality (test)',
    )

    trace4 = go.Scatter(
        x = XTrain[yTrain == 1,0],
        y = XTrain[yTrain == 1,1],
        mode = 'markers',
        marker = Marker(
            color = '#0000FF',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'high quality (train)'
    )

    trace5 = go.Scatter(
        x = XTrain[yTrain == 1,0],
        y = XTrain[yTrain == 1,1],
        mode = 'markers',
        marker = Marker(
            color = '#FF0000',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'low quality (train)'
    )

    layout = go.Layout(
        title = "2-Class Classification (k = %i, weights = '%s')" % (n_neighbors, weights),
        xaxis = dict(title = header[0]),
        yaxis = dict(title = header[1]),
        showlegend=True,
        autosize=False,
        width=700,
        height=500,
        margin=Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
        ),
    )

    data = [trace1, trace2, trace3, trace4, trace5]
    fig = dict(data=data, layout=layout)

    iplot(fig)


def dtDecisionPlot(XTrain, yTrain, XTest, yTest, header, max_depth=10):
    Xtrain = XTrain[:, :2]
    h = .02  # step size in the mesh

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(Xtrain, yTrain)

    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    # Z = Z.reshape(xx.shape)

    trace1 = go.Contour(
        x = np.arange(x_min, x_max, h),
        y = np.arange(y_min, y_max, h),
        z = Z.reshape(xx.shape),
        showscale=False,
        opacity=0.8,
        xaxis=header[0],
        yaxis=header[1],
        line = dict(
            width = 1,
            color = 'black'
        ),
        colorscale=[[0, '#AAAAFF'], [1, '#FFAAAA']],  # custom colorscale
    )

    trace2 = go.Scatter(
        x = XTest[yTest == 0,0],
        y = XTest[yTest == 0,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#0000FF',
            line = dict(
                width = 0.9,
            )
            #opacity=0.6
        ),
        name = 'high quality (test)'
    )

    trace3 = go.Scatter(
        x = XTest[yTest == 1,0],
        y = XTest[yTest == 1,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#FF0000',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
            #opacity=0.6
        ),
        name = 'low quality (test)',
    )

    trace4 = go.Scatter(
        x = XTrain[yTrain == 1,0],
        y = XTrain[yTrain == 1,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#0000FF',
            line = dict(
                width = 0.9,
            )
            #opacity=0.6
        ),
        name = 'high quality (train)'
    )

    trace5 = go.Scatter(
        x = XTrain[yTrain == 1,0],
        y = XTrain[yTrain == 1,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#FF0000',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
            #opacity=0.6
        ),
        name = 'low quality (train)'
    )

    layout = go.Layout(
        title = "2-Class classification Decision Trees",
        xaxis = dict(title = header[0]),
        yaxis = dict(title = header[1]),
        showlegend=True,
        autosize=False,
        width=700,
        height=500,
        margin=Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
        ),
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    fig = dict(data=data, layout=layout)
    iplot(fig)


def rfDecisionPlot(XTrain, yTrain, XTest, yTest, header, n_estimators=10):
    Xtrain = XTrain[:, :2]
    h = .02  # step size in the mesh

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
    clf.fit(Xtrain, yTrain)

    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    # Z = Z.reshape(xx.shape)

    trace1 = go.Contour(
        x = np.arange(x_min, x_max, h),
        y = np.arange(y_min, y_max, h),
        z = Z.reshape(xx.shape),
        showscale=False,
        opacity=0.8,
        xaxis=header[0],
        yaxis=header[1],
        line = dict(
            width = 1,
            color = 'black'
        ),
        colorscale=[[0, '#AAAAFF'], [1, '#FFAAAA']],  # custom colorscale
    )

    trace2 = go.Scatter(
        x = XTest[yTest == 0,0],
        y = XTest[yTest == 0,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#0000FF',
            line = dict(
                width = 0.9,
            )
            #opacity=0.6
        ),
        name = 'high quality (test)'
    )

    trace3 = go.Scatter(
        x = XTest[yTest == 1,0],
        y = XTest[yTest == 1,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#FF0000',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
            #opacity=0.6
        ),
        name = 'low quality (test)',
    )

    trace4 = go.Scatter(
        x = XTrain[yTrain == 1,0],
        y = XTrain[yTrain == 1,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#0000FF',
            line = dict(
                width = 0.9,
            )
            #opacity=0.6
        ),
        name = 'high quality (train)'
    )

    trace5 = go.Scatter(
        x = XTrain[yTrain == 1,0],
        y = XTrain[yTrain == 1,1],
        mode = 'markers',
        marker = Marker(
            #color = [('#0000FF' if i == 0 else '#FF0000') for i in yTest],
            color = '#FF0000',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
            #opacity=0.6
        ),
        name = 'low quality (train)'
    )

    layout = go.Layout(
        title = "2-Class classification Random Forests",
        xaxis = dict(title = header[0]),
        yaxis = dict(title = header[1]),
        showlegend=True,
        autosize=False,
        width=700,
        height=500,
        margin=Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    fig = dict(data=data, layout=layout)
    iplot(fig)


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

    trace = go.Scatter(
        y=avgPred,
        x=np.arange(df.shape[0]),
        mode='markers+lines',
        name = "Ensemble accuracy trend"
    )

    layout = go.Layout(
        title = "Ensemble accuracy over increasing number of trees",
        xaxis = dict(title = "Number of trees", nticks = 15),
        yaxis = dict(title = "Accuracy"),
        showlegend=False,
        autosize=False,
        width=1000,
        height=500,
        margin=Margin(
            l=70,
            r=50,
            b=100,
            t=50,
            pad=4
        ),
    )

    data = [trace]

    fig = dict(data=data, layout=layout)
    iplot(fig)
