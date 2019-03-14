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
# print(dfIris.tail())

# =============================
# separate df by its species

dfSetosa = dfIris[dfIris['target'] == 0]
# print(dfSetosa)
dfVersicolor = dfIris[dfIris['target'] == 1]
# print(dfVersicolor)
dfVirginica = dfIris[dfIris['target'] == 2]
print(dfVirginica)

# =============================
# scatter plot

fig = plt.figure('Iris Data', figsize=(14,7))

# plot sepal length vs sepal width
plt.subplot(121)
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
plt.title('Sepal length (cm) vs Sepal width (cm)')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.legend(['0 Setosa', '1 Versicolor', '2 Virginica'])
plt.grid(True)

# plot petal length vs petal width
plt.subplot(122)
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
plt.title('Petal length (cm) vs Petal width (cm)')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.legend(['0 Setosa', '1 Versicolor', '2 Virginica'])
plt.grid(True)

plt.show()