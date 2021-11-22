import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

dg = pd.read_csv('bank-additional-full.csv', sep = ';')
print(dg.head())
dg = dg.rename(columns =  
               {"y": "deposit"} 
               )
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dg['job'] = le.fit_transform(dg['job'])
dg['marital'] = le.fit_transform(dg['marital'])
dg['education'] = le.fit_transform(dg['education'])
dg['default'] = le.fit_transform(dg['default'])
dg['housing'] = le.fit_transform(dg['housing'])
dg['loan'] = le.fit_transform(dg['loan'])
dg['contact'] = le.fit_transform(dg['contact'])
dg['month'] = le.fit_transform(dg['month'])
dg['day_of_week'] = le.fit_transform(dg['day_of_week'])
dg['poutcome'] = le.fit_transform(dg['poutcome'])

dg['deposit'].replace({'no': 0, 'yes': 1},inplace = True)
dg['pdays'].replace({999: np.nan},inplace = True)
dg.pdays.fillna(6, inplace = True)

deposit0 = dg[dg['deposit'] == 0]
deposit1 = dg[dg['deposit'] == 1]
new_data = pd.concat([deposit0.sample(len(deposit1), random_state =0), deposit1], axis =0).reset_index(drop=True)
#month, age
X = new_data.drop(columns=['deposit','housing', 'loan','default', 'contact', 'campaign', 'day_of_week','pdays', 'cons.conf.idx','cons.price.idx','emp.var.rate', 'euribor3m','nr.employed','month'])
y = new_data['deposit']

print(X)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 3)
for train_indices, test_indices in skf.split(X,y):
    print(train_indices)
    print(test_indices)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

forest_clf = SVC(C =  0.7, gamma = 'auto', kernel = 'rbf')
forest_clf.fit(X[train_indices], y[train_indices])

pickle.dump(forest_clf, open('model_classifier.pkl', 'wb'))


