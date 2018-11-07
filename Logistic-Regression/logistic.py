from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats

features = pd.read_csv('../data/2015-01-28.csv')
le = preprocessing.LabelEncoder()

# preprocessing
features['income'] = features['income'].fillna(features['income'].mean())
features = features.drop('sp_index', axis=1)
features = features.drop('id', axis=1)
features = features.fillna('0')
# employee, gender, age, since_first_date, mortagage, shortdeposit, medium
features = features[['employee', 'gender', 'age', 'since_first_date', 'mortgage', 'shortdeposit', 'mediumdeposit']]

# Convert categorical variable to numeric
for i in ['employee','gender','res_index','for_idx','channel','cate']:
    features[i] = le.fit_transform(features[i])

print "The shape of our features is : (%d, %d)"%(features.shape[0], features.shape[1])

prediction_list = ["fund", "eaccount", "loans", "credit_card"]
prediction_index = 3
labels = np.array(features[prediction_list[prediction_index]])
features = features.drop(prediction_list[prediction_index],axis = 1 )
# features = pd.get_dummies(features)
feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size = 0.25,random_state = 42)

print "Training Features Shape: (%d, %d)"%(train_features.shape[0], train_features.shape[1])
print "Training Labels Shape: (%d)"%(train_labels.shape[0])
print "Testing Features Shape: (%d, %d)"%(test_features.shape[0], test_features.shape[1])
print "Testing Labels Shape: (%d)"%(test_labels.shape[0])

# sc = StandardScaler()
# sc.fit(train_features)
# train_features_std = sc.transform(train_features)
# test_features_std = sc.transform(test_features)

logistic = LogisticRegression(C=1000.0, random_state=0)
logistic.fit(train_features, train_labels)  
params = np.append(logistic.intercept_, logistic.coef_)
predictions = logistic.predict(test_features)
# predictions = logistic.predict(train_features)
errors = abs(predictions - test_labels)

# new_features = pd.DataFrame({"Constant": np.ones(len(train_features))}).join(pd.DataFrame(train_features))
# MSE = (sum(train_labels - predictions) ** 2) / (len(new_features) - len(new_features.columns))
# var_b = MSE * (np.linalg.inv(np.dot(new_features.T, new_features)).diagonal())
# sd_b = np.sqrt(var_b)
# ts_b = params / sd_b

# p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(new_features) - 1))) for i in ts_b]

# sd_b = np.round(sd_b, 3)
# ts_b = np.round(ts_b, 3)
# p_values = np.round(p_values, 3)
# params = np.round(params, 4)

# myDF3 = pd.DataFrame()
# myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
# print myDF3

mape = 100 * errors

accuracy = 100 - np.mean(mape)
print "'Accuracy(%s): %.2f'"%(prediction_list[prediction_index], round(accuracy, 2))