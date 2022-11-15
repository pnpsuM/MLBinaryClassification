import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# metadata
# "name" is unique(891)
# "ticket" and "cabin" contains duplicates
# "Cabin" too much blank(dropped)

# 1)Class
# higher class = more survival(Pclass=1, more survival)
# lower class = less survival(Pclass=3, less survival)

# 2)Sex
# females more survival, males less

# 3)Age(significant correlation)
# teenagers and children below 18 more survival
# elderly above 60 less survival

# 4)Parch
# If there are 1 or more

label_encoder_x= LabelEncoder()

# import datasets
dataset = pd.read_csv("datasets/train.csv")
dataset_test = pd.read_csv("datasets/test.csv")
dataset = dataset.drop(["Cabin","PassengerId"], axis=1)

# Explore data
print(dataset.describe(include=['O']))
print(dataset.describe())
pd.set_option('display.max_columns', None)

# deal with "Sex" column
# encode "Sex" into '0'(F) or '1'(M)
dataset['Sex'] = label_encoder_x.fit_transform(dataset['Sex'])

# deal with "Fare" column
# set "Fare" as individual price
dataset["Fare"] = (dataset["Fare"]/dataset.groupby("Ticket")["Fare"].transform("count")).astype('float')
dataset = dataset.drop(["Ticket"], axis=1)
# display "FareBand"
# print(dataset[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))
# cut "Fare" into 4 groups *
dataset['FareBand'] = pd.qcut(dataset['Fare'],4)
# encode "Fare" into 4 groups
dataset.loc[ dataset['Fare'] <= 7.762, 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.762) & (dataset['Fare'] <= 8.85), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 8.85) & (dataset['Fare'] <= 24.288), 'Fare'] = 2
dataset.loc[ dataset['Fare'] > 24.288, 'Fare'] = 3
dataset['Fare'] = dataset['Fare'].astype(int)
dataset = dataset.drop(["FareBand"],axis=1)

# deal with "Embarked" column
# fill in with the node of the data since only 2 missing row of data
dataset["Embarked"] = dataset["Embarked"].fillna(dataset.Embarked.dropna().mode()[0])
embarkedLabel = {"S": 1, "C": 2, "Q": 3}
dataset['Embarked'] = dataset['Embarked'].map(embarkedLabel)

# deal with "SibSp" and "Parch"(feature engineering)
# dataset["Alone"] = dataset["SibSp"] + dataset["Parch"]

# deal with "Name"(change into "title" and help with predicting age)
dataset["Title"] = dataset["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
# label minor affecting title as "Rare"
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
# same-meaning words corrections
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
# "Rev" is included because all of them didn't survive and there are  6 of them in training set *
titleLabel = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rev": 5,"Rare": 6}
dataset['Title'] = dataset['Title'].map(titleLabel)
dataset['Title'] = dataset['Title'].fillna(0)
print(dataset.head())

# heat map
# colormap = plt.cm.RdBu
# plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(dataset.astype(float).corr(),linewidths=0.1,vmax=1.0,
#             square=True, cmap=colormap, linecolor='white', annot=True)


# tools for checking data
# to check categorical data mean
# print(dataset[["Title", "Survived"]].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# # #
# # # # to check categorical data count
# print(dataset[["Title", "Survived"]].groupby(['Title']).size().reset_index(name='count').sort_values(by='count', ascending=False))


# target = np.array(dataset["Survived"])
# X = np.array(dataset[["Age"]])
# y = target.astype(int)
# print(y)
# model = LinearRegression()
# model.fit(X, y)
# predict = model.predict(X)
# plot = predict.flatten()
# print(model.score(X, predict))
# plt.scatter(X, y)
# # plt.plot(plot,y)
# plt.show()

# trim the cabin *
# dataset['Cabin'] = dataset.apply(lambda row: row['Cabin'][0:1], axis=1)


