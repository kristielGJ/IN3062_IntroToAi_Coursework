import os
import random
import pandas as pd

path = "."  #absolute or relative path to the folder containing the file. 
            #"." for current folder

filename_read = os.path.join(path, "train.csv")
df = pd.read_csv(filename_read)

filename_read_test = os.path.join(path, "test.csv")
df_t = pd.read_csv(filename_read_test)

df = df.select_dtypes(include=['int', 'float'])
df_t = df_t.select_dtypes(include=['int', 'float'])

#Zero Rule Algorithm:

#Predicting a class label
 
# zero rule algorithm for classification
def zeroRuleAlgo(train, test):
	outputVal = [row[-1] for row in train]
	p = max(set(outputVal), key=outputVal.count)
	p = [p for i in range(len(test))]
	return p
 
random.seed(1)
classValPrediction = zeroRuleAlgo(df, df_t)
print(classValPrediction)