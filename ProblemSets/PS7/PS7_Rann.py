# Import Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load in data
df = pd.read_csv('C:/Users/radle/DScourseS23/ProblemSets/PS7/wages.csv')

# Drop missing values
df_drop = df.dropna(subset=['hgc', 'tenure'])

# Fit linear regression models
model1 = sm.OLS(df_drop['logwage'], sm.add_constant(df_drop['hgc'])).fit()
model2 = sm.OLS(df_drop['logwage'], sm.add_constant(df_drop['tenure'])).fit()

# Create summary table
summary = summary_col([model1, model2], stars=True, float_format='%.2f',
                      model_names=['Model 1', 'Model 2'],
                      info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                 'R2':lambda x: "{:.2f}".format(x.rsquared)})
print(summary)

#impute missing values with mean
df_mean = df.fillna(df.mean())

#impute missing values with predicted values from complete cases regression
#first fit regression model on complete cases
complete_cases = df_drop[['logwage', 'hgc', 'college', 'tenure', 'tenure2', 'age', 'married']]
cc_model = sm.OLS(complete_cases['logwage'], sm.add_constant(complete_cases[['hgc', 'college', 'tenure', 'tenure2', 'age', 'married']])).fit()

#use model to predict missing values
df_pred = df.copy()
df_pred['logwage'] = cc_model.predict(sm.add_constant(df_pred[['hgc', 'college', 'tenure', 'tenure2', 'age', 'married']]))

#impute missing values using MICE
from impyute.imputation.cs import mice

#select relevant columns for MICE imputation
df_mice = df[['logwage', 'hgc', 'college', 'tenure', 'tenure2', 'age', 'married']]

#run MICE imputation
imputed_data = mice(df_mice.values)

#convert imputed data back to a pandas dataframe
df_mice_imputed = pd.DataFrame(imputed_data, columns=df_mice.columns)

#add imputed logwage column to original data
df_mice_imputed['logwage'] = df_mice_imputed['logwage'].astype(float)

#fit regression model on complete cases
model1 = sm.OLS(df_drop['logwage'], sm.add_constant(df_drop[['hgc', 'college', 'tenure', 'tenure2', 'age', 'married']])).fit()

#fit regression model on mean imputed data
model2 = sm.OLS(df_mean['logwage'], sm.add_constant(df_mean[['hgc', 'college', 'tenure', 'tenure2', 'age', 'married']])).fit()

#fit regression model on predicted values imputed data
model3 = sm.OLS(df_pred['logwage'], sm.add_constant(df_pred[['hgc', 'college', 'tenure', 'tenure2', 'age', 'married']])).fit()

#fit regression model on MICE imputed data
model4 = sm.OLS(df_mice_imputed['logwage'], sm.add_constant(df_mice_imputed[['hgc', 'college', 'tenure', 'tenure2', 'age', 'married']])).fit()

#combine regression models into one table
regression_models = pd.concat([model1.params, model2.params, model3.params, model4.params], axis=1)
regression_models.columns = ['Complete Cases', 'Mean Imputation', 'Predicted Values Imputation', 'MICE Imputation']
regression_models.index = ['Intercept', 'hgc', 'college', 'tenure', 'tenure2', 'age', 'married']
regression_models.index.name = "Model"

