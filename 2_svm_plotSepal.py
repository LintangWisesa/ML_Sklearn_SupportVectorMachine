import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))

# ==========================
# create dataframe

dfIris = pd.DataFrame(
    iris['data'],
    columns = iris['feature_names']
)
dfIris['target'] = iris['target']
dfIris['spesies'] = dfIris['target'].apply(
    lambda index: iris['target_names'][index]
)
print(dfIris.head())
print(dfIris.iloc[54:56])
print(dfIris.tail())

# =============================
# separate df by its species

dfSetosa = dfIris[dfIris['target'] == 0]
# print(dfSetosa)
dfVersicolor = dfIris[dfIris['target'] == 1]
# print(dfVersicolor)
dfVirginica = dfIris[dfIris['target'] == 2]
print(dfVirginica)

# ==========================
# split datasets: train 90% & test 10%

from sklearn.model_selection import train_test_split
xtra, xtes, ytra, ytes = train_test_split(
    dfIris[[
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]],
    dfIris['spesies'],
    test_size = .1
)

# print(len(xtra))
# print(len(xtes))

# ==========================
# support vector machine
# support vector classifier

from sklearn.svm import SVC
model = SVC(gamma = 'auto')

# train
model.fit(xtra, ytra)

# predict
# print(model.predict([[5.1, 3.5, 1.4, 0.2]]))
# print(model.predict([[5.7, 2.8, 4.5, 1.3]]))
# print(model.predict([[6.2, 3.4, 5.4, 2.3]]))

print(model.predict([xtes.iloc[0]]))
print(ytes.iloc[0])

# accuracy
print(model.score(xtes, ytes) * 100, '%')

# =============================
# plot svm

def bikin_meshgrid(x, y):
    x_min = x.min() - 1
    x_max = x.max() + 1
    y_min = y.min() - 1
    y_max = y.max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, .02),
        np.arange(y_min, y_max, .02)
    )
    return xx, yy

sepal = iris['data'][:, :2]
x0, x1 = sepal[:, 0], sepal[:, 1]
xx, yy = bikin_meshgrid(x0, x1)
# print(xx)
# print(yy)

model.fit(sepal, iris['target'])

# ==========================
# plot

def plotSVM(ax, model, xx, yy, **params):
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    hasil = ax.contourf(xx, yy, z, **params)
    return hasil

fig = plt.figure('SVM')
ax = plt.subplot()

plotSVM(ax, model, 
    xx, yy, 
    cmap = 'coolwarm', 
    alpha = .2
)
plt.scatter(
    dfSetosa['sepal length (cm)'],
    dfSetosa['sepal width (cm)'],
    color = 'r',
    marker = 'o'
)
plt.scatter(
    dfVersicolor['sepal length (cm)'],
    dfVersicolor['sepal width (cm)'],
    color = 'y',
    marker = 'o'
)
plt.scatter(
    dfVirginica['sepal length (cm)'],
    dfVirginica['sepal width (cm)'],
    color = 'b',
    marker = 'o'
)
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_title('Sepal length (cm) vs Sepal width (cm)')
plt.show()