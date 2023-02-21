#import dataframe package
import pandas as pd
#load file as pd
df = pd.read_json("dates.json")

#Look at df

print(df.head())

#Find df object
df_type = type(df)
print(df_type)
#print first 10 rows
print(df.head(10))
