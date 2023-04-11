import pandas as pd
import numpy as np
import random

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV





#Load in data set
pd.set_option('display.max_columns', None)

housing = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None, delim_whitespace=True)
housing.columns = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat",
                   "medv"]
print(housing.head())
#print(housing)


#Set seed
random.seed(123456)

#ML data sets

housing_split = train_test_split(housing, test_size=0.2)
housing_train = housing_split[0]
housing_test = housing_split[1]
print(housing_train.shape)

#Make recipe

ct = ColumnTransformer([
    ('log_transform', FunctionTransformer(func=np.log, validate=False), ['medv']),
    ('bin2factor', FunctionTransformer(func=pd.Series.astype, validate=False, kw_args={'dtype': 'category'}), ['chas']),
], remainder='passthrough')

recipe = Pipeline([
    ('column_transform', ct),
    ('interaction_terms', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('polynomial_features', PolynomialFeatures(degree=2, include_bias=False)),


])


# recipe = Pipeline([
#     ('log_transform', FunctionTransformer(func=np.log, validate=False), columns=['medv']),
#     ('bin2factor', FunctionTransformer(func=pd.Series.astype, validate=False, kw_args={'dtype': 'category'}), columns=['chas']),
#     ('interaction_terms', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
#     ('polynomial_features', PolynomialFeatures(degree=6, include_bias=False))

# ])

# Juice it
housing_train_prep = recipe.fit_transform(housing_train)

housing_test_prep = recipe.transform(housing_test)

#Create x and y training data

housing_train_x = pd.DataFrame(housing_train_prep).iloc[:, :-1]
housing_test_x = pd.DataFrame(housing_test_prep).iloc[:, :-1]
housing_train_y = pd.DataFrame(housing_train_prep).iloc[:, -1]
housing_test_y = pd.DataFrame(housing_test_prep).iloc[:, -1]

#LASSO Model

lasso = LassoCV(cv=6, random_state=123456)

lasso.fit(housing_train_x, housing_train_y)

# print("LASSO Coefficients:")
# for name, coef in zip(list(housing_train_x.columns), list(lasso.coef_)):
#     print(name, ':', coef)

lasso_pred = lasso.predict(housing_test_x)

#Optimal lambda

alphas = lasso.alphas_
cv_results = lasso.mse_path_
mean_cv_scores = np.mean(cv_results, axis=1)
best_alpha_idx = np.argmax(mean_cv_scores)
best_alpha = alphas[best_alpha_idx]

print(f"The optimal value of alpha is {best_alpha:.3f}.")

#RMSE in Sample

lasso_pred_train = lasso.predict(housing_train_x)
in_sample_rmse = np.sqrt(mean_squared_error(housing_train_y, lasso_pred_train))
print('In-sample RMSE:', in_sample_rmse)

#RMSE out of sample

out_of_sample_rmse = np.sqrt(mean_squared_error(housing_test_y, lasso_pred))
print('Out-of-sample RMSE:', out_of_sample_rmse)


#With the ridge model

ridge = RidgeCV(cv=6)
ridge.fit(housing_train_x, housing_train_y)

print("Optimal value of l:", ridge.alpha_)

#In sample RMSE
ridge_train_pred = ridge.predict(housing_train_x)
ridge_train_rmse = np.sqrt(mean_squared_error(housing_train_y, ridge_train_pred))
print("In-sample RMSE:", ridge_train_rmse)


#Out of sample RMSE
ridge_test_pred = ridge.predict(housing_test_x)
ridge_test_rmse = np.sqrt(mean_squared_error(housing_test_y, ridge_test_pred))
print("Out-of-sample RMSE:", ridge_test_rmse)
