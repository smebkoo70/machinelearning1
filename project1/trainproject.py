import numpy

from sklearn import datasets
import pandas as pd
import warnings


warnings.filterwarnings("ignore")

iris = datasets.load_iris()

# anchor_ratio,trans_range,node_density,iterations,ale,sd_ale,resale
iris_data = pd.DataFrame(iris.data,columns=['anchor_ratio','trans_range','node_density','iterations'])
iris_data['resale'] = iris.target
iris_data.to_csv('mcs_ds_edited_iter_shuffled1.csv',index=None)

# https://archive.ics.uci.edu/ml/index.php
iris = pd.read_csv('mcs_ds_edited_iter_shuffled.csv')

X = iris[['anchor_ratio','trans_range','node_density','iterations']]
Y = iris['resale']


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3
)  # 第一个参数：特征   第二个参数：标签   test_size：训练集与测试集的比值


# 逻辑回归
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
# 分类器与数据拟合
classifier.fit(X_train, Y_train)


# 神经网络 （注意：仅适用于 0.18 或更高版本的 scikit-learn）

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train, Y_train)



data = pd.DataFrame()
data['真实值'] = Y_test
data['预测值'] = classifier.predict(X_test)

print(data)

from sklearn.metrics import classification_report
print(classification_report(data['真实值'],data['预测值']))

import matplotlib.pyplot as plt
plt.scatter(range(len(X_test)),data['真实值'])    # 画散点图
plt.plot(range(len(X_test)),data['预测值'],c='g') # 画折线图
plt.show()