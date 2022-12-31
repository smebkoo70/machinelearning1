import pandas as pd
from sklearn import datasets

import warnings


warnings.filterwarnings("ignore")

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_Y = loaded_data.target
data = pd.DataFrame(data_X,columns=loaded_data.feature_names)
data['房价值'] = data_Y
data.to_csv('house_data1.csv',index=None)


data = pd.read_csv('house_data.csv')

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3
)  # 第一个参数：特征   第二个参数：标签   test_size：训练集与测试集的比值

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,Y_train)


data = pd.DataFrame()
data['真实值'] = Y_test
data['预测值'] = lr.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(range(len(X_test)),data['真实值'])    # 画散点图
plt.plot(range(len(X_test)),data['预测值'],c='g') # 画折线图
plt.show()


print(data)

print('模型评分：'+str(lr.score(X_test,Y_test)))
