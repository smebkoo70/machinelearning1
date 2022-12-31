import pandas as pd
from sklearn import datasets

import warnings

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# from sklearn import datasets
# import pandas as pd
# iris =datasets.load_iris()
# iris_X = iris.data    # 鸢尾花特征
# iris_Y = iris.target  # 鸢尾花标签
# 直接用上面两个训练也没有问题，本教程将它们整理成表格，更贴近现实。
# iris_data = pd.DataFrame(iris.data,columns=['花萼长度','花萼宽度','花瓣长度','花瓣宽度'])
# iris_data['种类'] = iris.target
# iris_data.to_excel('鸢尾花数据.xlsx',index=None)

# loaded_data = datasets.load_boston()
# data_X = loaded_data.data
# data_Y = loaded_data.target
# data = pd.DataFrame(data_X, columns=loaded_data.feature_names)
# data['ale'] = data_Y
# data.to_csv('mcs_ds_edited_iter_shuffled1.csv', index=None)

data = pd.read_csv('mcs_ds_edited_iter_shuffled.csv')

# X = data.iloc[:, :-1]
X = data[['anchor_ratio','trans_range','node_density','iterations']]
Y = data['ale']
# Y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3
)  # 第一个参数：特征   第二个参数：标签   test_size：训练集与测试集的比值

from sklearn.linear_model import LinearRegression, LogisticRegression

lr = LinearRegression()
# lr = DecisionTreeRegressor()
# lr = SVR()
# lr = KNeighborsRegressor()
# lr = MLPRegressor()
lr.fit(X_train.astype('int'), Y_train)

data = pd.DataFrame()
data['真实值'] = Y_test
data['预测值'] = lr.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(range(len(X_test)), data['真实值'])  # 画散点图
plt.plot(range(len(X_test)), data['预测值'], c='g')  # 画折线图
plt.show()

print(data)

print('模型评分：' + str(lr.score(X_test, Y_test)))
