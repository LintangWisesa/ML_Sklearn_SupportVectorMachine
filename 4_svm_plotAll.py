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
model1 = SVC(gamma = 'auto')
model2 = SVC(gamma = 'auto')

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
petal = iris['data'][:, 2:]
x2, x3 = petal[:, 0], petal[:, 1]
ww, xx = bikin_meshgrid(x0, x1)
yy, zz = bikin_meshgrid(x2, x3)

model1.fit(sepal, iris['target'])
model2.fit(petal, iris['target'])

# ==========================
# plot

def plotSVM(ax, model, xx, yy, **params):
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    hasil = ax.contourf(xx, yy, z, **params)
    return hasil

fig = plt.figure('SVM', figsize=(14,7))

ax = plt.subplot(121)
plotSVM(ax, model1, 
    ww, xx, 
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
ax.set_title('Sepal length (cm) vs Petal width (cm)')

ax = plt.subplot(122)
plotSVM(ax, model2, 
    yy, zz, 
    cmap = 'coolwarm', 
    alpha = .2
)
plt.scatter(
    dfSetosa['petal length (cm)'],
    dfSetosa['petal width (cm)'],
    color = 'r',
    marker = 'o'
)
plt.scatter(
    dfVersicolor['petal length (cm)'],
    dfVersicolor['petal width (cm)'],
    color = 'y',
    marker = 'o'
)
plt.scatter(
    dfVirginica['petal length (cm)'],
    dfVirginica['petal width (cm)'],
    color = 'b',
    marker = 'o'
)
ax.set_xlabel('Petal length (cm)')
ax.set_ylabel('Petal width (cm)')
ax.set_title('Petal length (cm) vs Petal width (cm)')

plt.show()