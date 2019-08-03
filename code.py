import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# collecting the data
df = pd.read_csv("UCI_Credit_Card.csv")

# combine the 'other' category into one label 
df['MARRIAGE'] = df['MARRIAGE'].replace(0,3)
df['EDUCATION'] = df['EDUCATION'].replace([0,5,6],4)

# Feature Selection
# pearson correlation to find the multicolinearity in each feature, but only focus on PAY_X, BILL_AMTX, and PAY_AMTX
pay = df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
      'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]

# Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = pay.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# from the pearson corr table, PAY_X can be said has multicolinearity each other. BILL_AMTX is the same as well
# decided to only use 1 feature for each of them

df = df.drop(['PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis=1)

# in this case, assume the payment delay gather into one category (1)
# some labels have -2, assume that it the same as duly payment -1
df['PAY_0'] = df['PAY_0'].replace([2,3,4,5,6,7,8],1)
df['PAY_0'] = df['PAY_0'].replace(-2,-1)

# create the one hot encoding for category variable
cat_feat = ["EDUCATION","MARRIAGE","PAY_0"]
df_dummy = pd.get_dummies(df,columns=cat_feat, drop_first=True)

# Modelling the data
# define the X and Y value
X = df_dummy.drop(['ID','default.payment.next.month'],axis=1)
Y = df_dummy['default.payment.next.month']

# scaling the X variable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split into train and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

# start to modelling the SVM model with linear kernel. Retrieved the code from below link.
# https://www.kaggle.com/ainslie/credit-card-default-prediction-analysis

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
classifier1 = SVC(kernel="linear")
classifier1.fit( X_train, y_train )
y_pred1 = classifier1.predict( X_test )

cm1 = confusion_matrix( y_test, y_pred1 )
print("Accuracy on Test Set for kernel-SVM = %.2f" % ((cm1[0,0] + cm1[1,1] )/len(X_test)))
scoresSVC = cross_val_score( classifier1, X_train, y_train, cv=10)
print("Mean kernel-SVM CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresSVC.mean(), scoresSVC.std() ))

# modelling KNeighbors, number of neighbor = 2
from sklearn.neighbors import KNeighborsClassifier
classifier3 = KNeighborsClassifier(n_neighbors=2)
classifier3.fit( X_train, y_train )
y_pred3 = classifier3.predict( X_test )
cm3 = confusion_matrix( y_test, y_pred3 )
acc3 = (cm3[0,0] + cm3[1,1] )/len(X_test)
print("Accuracy on Test Set for KNeighborsClassifier = %.2f" % ((cm3[0,0] + cm3[1,1] )/len(X_test)))
scoresKN = cross_val_score( classifier3, X_train, y_train, cv=10)
print("Mean KN CrossVal Accuracy on Train Set Set %.2f, with std=%.2f" % (scoresKN.mean(), scoresKN.std() ))
