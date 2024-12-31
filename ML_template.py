import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Set the random seed
np.random.seed(42)


# Paths to the data
test_path = 'data/test.csv'
train_path = 'data/train.csv'


# Load the data
X_test = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)


target = 'SalePrice'

# Split the data into X and y
X_train = train_data.drop(target, axis=1)
y_train = train_data[target]

#numerical and categorical columns
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
low_card_cols = [col for col in X_train.columns if X_train[col].nunique() < 10 and X_train[col].dtype == 'object']

#columns to keep and final data
cols = num_cols + low_card_cols
X_train = X_train[cols]
X_test = X_test[cols]

#splitting the data
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, test_size=0.2, random_state=42)  

#preprocessing the data
imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_valid[num_cols] = imputer.transform(X_valid[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

#scaling the data
scaler = StandardScaler()
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_valid[num_cols] = scaler.transform(X_valid[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_train_full = pd.concat([X_train, X_valid])
y_train_full = pd.concat([y_train, y_valid])

#random forest model
rfmodel = RandomForestRegressor(n_estimators=100, random_state=42,criterion='mae')
baserf = cross_val_score(rfmodel, X_train_full, y_train_full, cv=5, scoring='neg_mean_absolute_error')
print(f'Random Forest base MAE: {-1*baserf.mean()}')

#xgboost model
xgmodel = XGBRegressor(n_estimators=1000, learning_rate=0.05)
basexgb = cross_val_score(xgmodel, X_train_full, y_train_full, cv=5, scoring='neg_mean_absolute_error')
print(f'XGBoost base MAE: {-1*basexgb.mean()}')


#xgboost model with early stopping
modelxgb = XGBRegressor(n_estimators=1000, learning_rate=0.1)
modelxgb.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

predsxgb = modelxgb.predict(X_valid)
maexgb = mean_absolute_error(y_valid, predsxgb)
print(f'XGBoost MAE: {maexgb}')


###feature engineering

#mutual information
mi_scores = mutual_info_regression(X_train_full, y_train_full)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train_full.columns)
mi_scores = mi_scores.sort_values(ascending=True)

#get top 5 columns name with highest mutual information
top5 = mi_scores.index[-5:]

#plotting
plt.figure(dpi=100, figsize=(10, 5))
mi_scores.plot.bar(color='lightblue')
plt.title('Mutual Information Scores')
width = np.arange(len(mi_scores))
ticks = list(mi_scores.index)
plt.barh(width, mi_scores)
plt.yticks(width, ticks)
plt.show()

#create custom features
# by multiplying adding or taking ratio diving
# differnt features with each other


# Create principal components
pca_data = X_train_full[top5]
pca = PCA()
X_pca = pca.fit_transform(pca_data)
component =[f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component)

loading = pd.DataFrame(pca.components_.T, columns=component, index=pca_data.columns)
print(loading)
#use loading to interpret the principal components and create custom features
#by considering the loading value contrast between the features
#do with most differnce in loading value