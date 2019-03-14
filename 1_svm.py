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