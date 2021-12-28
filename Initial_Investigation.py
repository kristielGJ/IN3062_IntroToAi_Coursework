#
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


path = "."  #absolute or relative path to the folder containing the file. 
            #"." for current folder

filename_read = os.path.join(path, "train.csv")
df = pd.read_csv(filename_read)

'''Start of analysis of data'''
# Strip non-numerics
df = df.select_dtypes(include=['int', 'float'])

headers = list(df.columns.values)
fields = []
# Perform basic statistics (mean, variance, standard deviation, z scores) on a dataframe.

for field in headers:
    fields.append({
        'name' : field,
        'mean': df[field].mean(),
        'var': df[field].var(),
        'sdev': df[field].std(),# how dispersed the data is in relation to the mean

    })
#details on the data in numbers 
for col in df.columns:
    print(col + '\n_____________')
    print(df[col].value_counts())
    print('_____________________________\n')
    
#display statistics 
for field in fields:
    print(field)

#visualise data so that it is clear what the data is showing
df.hist(figsize=(20,20))
plt.show()
#sns.pairplot(df,hue='price_range')

#heat map isa correlation matrix used to show the correlations between each field 
plt.figure(figsize=(16,16))
sns.heatmap(df.corr(), annot=True, fmt=".2f");
'''End of analysis of data'''
