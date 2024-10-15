import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns

dataset = pd.read_csv('pulsar_data_train.csv')
datatest = pd.read_csv('pulsar_data_test.csv')

print(dataset.head())
print(dataset.describe())

n = dataset['target_class'].value_counts()
print(n)
n.plot(kind='bar')
plt.title('Bar plot for Target Class')
plt.show()
"""
n = dataset.iloc[:,2]
print(n)
n.plot(kind='line')
plt.show()
"""
for column in range(len(dataset.columns) - 1):
    sns.boxplot(dataset.iloc[:, column])
    plt.title('Box Plot')
    plt.xlabel('Values')
    plt.ylabel('Boxplot')
    plt.show()

    sns.violinplot(dataset.iloc[:, column])
    plt.title('Violin Plot')
    plt.xlabel('Values')
    plt.ylabel('Violin plot')
    plt.show()

selected_columns = dataset.columns[:-1]
selected_df = dataset[selected_columns]
numeric_cols = selected_df.select_dtypes(include=['number']).columns
numeric_df = selected_df[numeric_cols]
plt.figure(figsize=(7.5, 7.5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap')
plt.show()

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

null = round(100*(dataset.isnull().sum())/len(dataset), 2)
print(null)
# null = round(100*(datatest.isnull().sum())/len(datatest), 2)
# print(null)
cols = [2, 5, 7]
X_sub = X[:, cols]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_sub)
X_sub = imputer.transform(X_sub)
X = np.delete(X, cols, axis=1)
X = np.column_stack((X, X_sub))

X_df = pd.DataFrame(X)
null = round(100*(X_df.isnull().sum())/len(dataset), 2)
print(null)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
print(pd.DataFrame(y_pred).value_counts())
print(accuracy_score(y_pred, Y_test))
cm = confusion_matrix(y_pred, Y_test)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
y_pred = neigh.predict(X_test)
print(pd.DataFrame(y_pred).value_counts())
print(accuracy_score(y_pred, Y_test))
cm = confusion_matrix(y_pred, Y_test)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtc.fit(X_train, Y_train)
y_pred = dtc.predict(X_test)
print(pd.DataFrame(y_pred).value_counts())
print(accuracy_score(y_pred, Y_test))
cm = confusion_matrix(y_pred, Y_test)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, Y_train)
y_pred = gbc.predict(X_test)
print(pd.DataFrame(y_pred).value_counts())
print(accuracy_score(y_pred, Y_test))
cm = confusion_matrix(y_pred, Y_test)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()